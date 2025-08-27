"""
Nano-indentation simulation with Lagrangian augmented contact method
Supports Berkovich and spherical indenters with crystal plasticity
"""

import jax
import jax.numpy as np
import numpy as onp
import os
import glob
import matplotlib.pyplot as plt
import time
import meshio
import argparse
import logging
import subprocess
from functools import partial
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, get_meshio_cell_type
from jax_fem.utils import save_sol
from jax_fem import logger

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from models_cuivre import CrystalPlasticityCuivre, get_rot_mat_vmap

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class NeperMeshGenerator:
    """Generate polycrystalline meshes with Neper"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_polycrystal(self, num_grains, domain_size, mesh_size='medium'):
        """Generate polycrystalline mesh"""
        rcl_values = {'fine': 0.8, 'medium': 1.0, 'coarse': 1.5}
        rcl = rcl_values.get(mesh_size, 1.0)
        
        base_name = f"domain_{num_grains}grains_{mesh_size}"
        tess_file = os.path.join(self.output_dir, f"{base_name}.tess")
        msh_file = os.path.join(self.output_dir, f"{base_name}.msh")
        
        domain_str = f"cube({domain_size[0]},{domain_size[1]},{domain_size[2]})"
        
        cmd_tess = [
            "neper", "-T", "-n", str(num_grains), "-id", "0",
            "-regularization", "1", "-domain", domain_str,
            "-format", "tess,ori", "-o", os.path.join(self.output_dir, base_name)
        ]
        
        try:
            subprocess.run(cmd_tess, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Neper tessellation failed: {e.stderr}")
        
        cmd_mesh = [
            "neper", "-M", tess_file, "-elttype", "hex", "-order", "1",
            "-rcl", str(rcl), "-format", "msh",
            "-o", os.path.join(self.output_dir, base_name)
        ]
        
        try:
            subprocess.run(cmd_mesh, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Neper meshing failed: {e.stderr}")
        
        quaternions = self._generate_random_quaternions(num_grains)
        quat_file = os.path.join(self.output_dir, f"{base_name}_quat.txt")
        onp.savetxt(quat_file, quaternions, fmt='%.10f')
        
        return msh_file, quat_file
    
    def _generate_random_quaternions(self, n):
        """Generate n random unit quaternions"""
        u = onp.random.rand(n, 3)
        q = onp.zeros((n, 4))
        q[:, 0] = onp.sqrt(1 - u[:, 0]) * onp.sin(2 * onp.pi * u[:, 1])
        q[:, 1] = onp.sqrt(1 - u[:, 0]) * onp.cos(2 * onp.pi * u[:, 1])
        q[:, 2] = onp.sqrt(u[:, 0]) * onp.sin(2 * onp.pi * u[:, 2])
        q[:, 3] = onp.sqrt(u[:, 0]) * onp.cos(2 * onp.pi * u[:, 2])
        return q


class BerkovichIndenter:
    """Berkovich pyramidal indenter"""
    
    def __init__(self, tip_position):
        self.tip_position = onp.array(tip_position, dtype=onp.float64)
        self.half_angle_deg = 70.3
        self.half_angle_rad = onp.radians(self.half_angle_deg)
        self.tan_half_angle = onp.tan(self.half_angle_rad)
        self.cos_half_angle = onp.cos(self.half_angle_rad)
        self.sin_half_angle = onp.sin(self.half_angle_rad)
        self.indenter_type = 'berkovich'
    
    def compute_gap_and_normal(self, point_coords):
        """Compute gap and normal for Berkovich indenter"""
        x, y, z = point_coords
        tip_x, tip_y, tip_z = self.tip_position
        
        dx = x - tip_x
        dy = y - tip_y
        r = onp.sqrt(dx**2 + dy**2)
        
        z_cone_surface = tip_z + r / self.tan_half_angle
        gap = z - z_cone_surface
        
        if r > 1e-12:
            normal = onp.array([
                dx/r * self.sin_half_angle,
                dy/r * self.sin_half_angle,
                self.cos_half_angle
            ])
        else:
            normal = onp.array([0., 0., 1.])
        
        normal = normal / onp.linalg.norm(normal)
        return float(gap), normal, True


class SphericalIndenter:
    """Spherical indenter"""
    
    def __init__(self, tip_position, radius=1e-6):
        self.tip_position = onp.array(tip_position, dtype=onp.float64)
        self.radius = radius
        self.indenter_type = 'spherical'
    
    def compute_gap_and_normal(self, point_coords):
        """Compute gap and normal for spherical indenter"""
        point = onp.array(point_coords)
        center = self.tip_position + onp.array([0., 0., self.radius])
        
        vec_to_point = point - center
        distance_to_center = onp.linalg.norm(vec_to_point)
        
        gap = distance_to_center - self.radius
        
        if distance_to_center > 1e-12:
            normal = vec_to_point / distance_to_center
        else:
            normal = onp.array([0., 0., 1.])
        
        return float(gap), normal, True


class ContactDetector:
    """Contact detection for indentation"""
    
    def __init__(self, indenter, surface_node_ids, tolerance=1e-9):
        self.indenter = indenter
        self.surface_node_ids = surface_node_ids
        self.tolerance = tolerance
    
    def detect_contacts(self, mesh_points, displacement_field):
        """Detect contact nodes"""
        contacts = []
        deformed_positions = mesh_points[self.surface_node_ids] + displacement_field[self.surface_node_ids]
        
        for i, node_id in enumerate(self.surface_node_ids):
            point_coords = deformed_positions[i]
            gap, normal, in_zone = self.indenter.compute_gap_and_normal(point_coords)
            
            if gap <= self.tolerance and in_zone:
                contacts.append({
                    'node_id': int(node_id),
                    'gap': gap,
                    'normal': normal,
                    'position': point_coords
                })
        
        return contacts


class NanoIndentationContactProblem(CrystalPlasticityCuivre):
    """Nano-indentation problem with augmented Lagrangian contact"""
    
    def __init__(self, mesh, quat, cell_ori_inds, indenter, contact_detector, 
                 contact_params, indentation_depth=0.0, previous_vars=None):
        
        self.original_mesh = mesh
        self.indenter = indenter
        self.contact_detector = contact_detector
        self.contact_params = contact_params
        self.indentation_depth = indentation_depth
        self.lambda_contact = {}
        self.current_contacts = []
        
        points = mesh.points
        z_min = onp.min(points[:, 2])
        z_max = onp.max(points[:, 2])
        self.z_min = z_min
        self.z_max = z_max
        
        boundary_conditions = self._create_boundary_conditions(z_min)
        
        super().__init__(
            mesh=mesh, vec=3, dim=3, ele_type='HEX8',
            dirichlet_bc_info=boundary_conditions,
            additional_info=(quat, cell_ori_inds)
        )
        
        if previous_vars is not None:
            self.internal_vars = previous_vars[:4]
            if len(previous_vars) > 4:
                self.lambda_contact = previous_vars[4]
        
        self.dt = 0.01
    
    def _create_boundary_conditions(self, z_min):
        """Create boundary conditions (fixed bottom)"""
        def bottom_face(point):
            return np.isclose(point[2], z_min, atol=1e-6)
        
        def zero_displacement(point):
            return 0.0
        
        return [
            [bottom_face, bottom_face, bottom_face],
            [0, 1, 2],
            [zero_displacement, zero_displacement, zero_displacement]
        ]
    
    def compute_residual(self, sol_list):
        """Compute residual with augmented Lagrangian contact forces"""
        base_residual_list = super().compute_residual(sol_list)
        
        if len(sol_list) > 0:
            displacement_field = sol_list[0]
            
            self.current_contacts = self.contact_detector.detect_contacts(
                self.original_mesh.points, onp.array(displacement_field)
            )
            
            if self.current_contacts:
                contact_forces = self._compute_contact_forces_augmented()
                base_residual_list[0] = base_residual_list[0] - contact_forces
        
        return base_residual_list
    
    def _compute_contact_forces_augmented(self):
        """Compute augmented Lagrangian contact forces"""
        contact_forces = np.zeros((self.fes[0].num_total_nodes, self.fes[0].vec))
        epsilon_N = self.contact_params.get('penalty_parameter', 1e10)
        
        for contact in self.current_contacts:
            node_id = contact['node_id']
            gap = contact['gap']
            normal = np.array(contact['normal'])
            
            lambda_n = self.lambda_contact.get(node_id, 0.0)
            force_magnitude = lambda_n + epsilon_N * abs(gap)
            
            if gap < 0:
                force_vector = force_magnitude * (-normal)
                contact_forces = contact_forces.at[node_id].add(force_vector)
        
        return contact_forces
    
    def update_lagrange_multipliers(self):
        """Update Lagrange multipliers"""
        epsilon_N = self.contact_params.get('penalty_parameter', 1e10)
        updated_lambdas = {}
        
        for contact in self.current_contacts:
            node_id = contact['node_id']
            gap = contact['gap']
            
            lambda_old = self.lambda_contact.get(node_id, 0.0)
            lambda_new = max(0.0, lambda_old + epsilon_N * abs(gap))
            
            if abs(lambda_new) > 1e-12:
                updated_lambdas[node_id] = lambda_new
        
        lambda_change = 0.0
        for node_id in set(list(self.lambda_contact.keys()) + list(updated_lambdas.keys())):
            old_val = self.lambda_contact.get(node_id, 0.0)
            new_val = updated_lambdas.get(node_id, 0.0)
            lambda_change = max(lambda_change, abs(new_val - old_val))
        
        self.lambda_contact = updated_lambdas
        return lambda_change
    
    def get_internal_vars_with_lambda(self):
        """Return all internal variables including Lagrange multipliers"""
        return self.internal_vars + [self.lambda_contact]
    
    def get_contact_info(self):
        """Get contact state information"""
        if not self.current_contacts:
            return {
                'num_contacts': 0, 'total_force': 0.0, 'max_penetration': 0.0,
                'contact_area': 0.0, 'mean_pressure': 0.0, 'max_lambda': 0.0
            }
        
        total_force = 0.0
        max_penetration = 0.0
        max_lambda = 0.0
        epsilon_N = self.contact_params.get('penalty_parameter', 1e10)
        
        for contact in self.current_contacts:
            node_id = contact['node_id']
            gap = contact['gap']
            lambda_n = self.lambda_contact.get(node_id, 0.0)
            
            force_magnitude = lambda_n + epsilon_N * abs(gap)
            total_force += force_magnitude
            
            if gap < 0:
                max_penetration = max(max_penetration, abs(gap))
            
            max_lambda = max(max_lambda, lambda_n)
        
        num_contacts = len(self.current_contacts)
        if num_contacts > 0:
            element_size = onp.mean([
                onp.linalg.norm(self.original_mesh.points[i] - self.original_mesh.points[j])
                for i, j in zip(self.original_mesh.cells[:10, 0], self.original_mesh.cells[:10, 1])
            ])
            contact_area = num_contacts * (element_size/2)**2
            mean_pressure = total_force / contact_area if contact_area > 0 else 0
        else:
            contact_area = 0.0
            mean_pressure = 0.0
        
        return {
            'num_contacts': num_contacts, 'total_force': total_force,
            'max_penetration': max_penetration, 'contact_area': contact_area,
            'mean_pressure': mean_pressure, 'max_lambda': max_lambda
        }


class AugmentedLagrangianSolver:
    """Augmented Lagrangian solver using Uzawa algorithm"""
    
    def __init__(self, problem, max_uzawa_iter=50, uzawa_tol=1e-6):
        self.problem = problem
        self.max_uzawa_iter = max_uzawa_iter
        self.uzawa_tol = uzawa_tol
    
    def solve(self, initial_guess=None):
        """Solve using Uzawa iterations"""
        if initial_guess is None:
            sol_list = [onp.zeros((self.problem.fes[0].num_total_nodes,
                                   self.problem.fes[0].vec))]
        else:
            sol_list = initial_guess
        
        converged = False
        
        for uzawa_iter in range(self.max_uzawa_iter):
            try:
                solver_options = {
                    'initial_guess': sol_list,
                    'tol': 1e-8,
                    'rel_tol': 1e-10
                }
                sol_list = solver(self.problem, solver_options)
            except Exception as e:
                logger.error(f"Newton solver error: {e}")
                break
            
            contact_info = self.problem.get_contact_info()
            lambda_change = self.problem.update_lagrange_multipliers()
            
            max_gap = contact_info['max_penetration']
            
            if max_gap < self.uzawa_tol and lambda_change < self.uzawa_tol * 1e3:
                converged = True
                break
        
        return sol_list, converged


class NanoIndentationSimulation:
    """Main nano-indentation simulation class"""
    
    def __init__(self, output_base_dir="output"):
        self.output_base_dir = os.path.join(current_dir, output_base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()
    
    def setup_directories(self):
        """Create output directory structure"""
        self.output_dir = os.path.join(self.output_base_dir, f"simulation_{self.timestamp}")
        self.mesh_dir = os.path.join(self.output_dir, "mesh")
        self.vtk_dir = os.path.join(self.output_dir, "vtk")
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.data_dir = os.path.join(self.output_dir, "data")
        
        for directory in [self.mesh_dir, self.vtk_dir, self.figures_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def run_simulation(self, simulation_params):
        """Run complete nano-indentation simulation"""
        logger.info("Starting nano-indentation simulation with augmented Lagrangian contact")
        logger.info(f"Backend: {jax.lib.xla_bridge.get_backend().platform}")
        
        # Generate mesh
        mesh_generator = NeperMeshGenerator(self.mesh_dir)
        msh_file, quat_file = mesh_generator.generate_polycrystal(
            simulation_params['num_grains'],
            simulation_params['domain_size'],
            simulation_params['mesh_size']
        )
        
        # Load mesh and orientations
        mesh, quat, cell_ori_inds = self.load_mesh_and_orientations(msh_file, quat_file)
        
        # Create indenter
        points = mesh.points
        center_x = onp.mean(points[:, 0])
        center_y = onp.mean(points[:, 1])
        max_z = onp.max(points[:, 2])
        
        initial_tip_position = [center_x, center_y, max_z]
        
        if simulation_params['indenter_type'] == 'berkovich':
            indenter = BerkovichIndenter(initial_tip_position)
        elif simulation_params['indenter_type'] == 'spherical':
            radius = simulation_params.get('sphere_radius', 1e-6)
            indenter = SphericalIndenter(initial_tip_position, radius)
        else:
            raise ValueError(f"Unknown indenter type: {simulation_params['indenter_type']}")
        
        # Contact detector
        surface_node_ids = onp.where(onp.isclose(points[:, 2], max_z, atol=1e-6))[0]
        contact_detector = ContactDetector(
            indenter, surface_node_ids,
            tolerance=simulation_params.get('contact_tolerance', 1e-9)
        )
        
        # Contact parameters
        contact_params = {
            'penalty_parameter': simulation_params.get('penalty_parameter', 1e10),
            'tolerance': simulation_params.get('contact_tolerance', 1e-9)
        }
        
        # Incremental simulation
        max_depth = simulation_params['max_indentation_depth']
        num_steps = simulation_params['num_steps']
        depth_increments = onp.linspace(0, max_depth, num_steps + 1)[1:]
        
        results = {
            'depths': [], 'forces': [], 'num_contacts': [], 'max_penetrations': [],
            'contact_areas': [], 'mean_pressures': [], 'max_displacements': [],
            'computation_times': [], 'max_lambdas': []
        }
        
        previous_vars = None
        previous_solution = None
        
        for step, target_depth in enumerate(depth_increments):
            logger.info(f"Step {step+1}/{num_steps}: depth = {target_depth*1e9:.1f} nm")
            
            # Move indenter
            current_indenter_position = [center_x, center_y, max_z - target_depth]
            indenter.tip_position = onp.array(current_indenter_position)
            
            # Create problem
            problem = NanoIndentationContactProblem(
                mesh=mesh, quat=quat, cell_ori_inds=cell_ori_inds,
                indenter=indenter, contact_detector=contact_detector,
                contact_params=contact_params, indentation_depth=target_depth,
                previous_vars=previous_vars
            )
            
            # Solve with augmented Lagrangian
            uzawa_solver = AugmentedLagrangianSolver(
                problem,
                max_uzawa_iter=simulation_params.get('max_uzawa_iter', 50),
                uzawa_tol=simulation_params.get('uzawa_tolerance', 1e-6)
            )
            
            start_time = time.time()
            sol_list, converged = uzawa_solver.solve(initial_guess=previous_solution)
            solve_time = time.time() - start_time
            
            if not converged:
                logger.warning(f"Uzawa did not converge at step {step+1}")
            
            # Analyze results
            max_displacement = onp.max(onp.abs(sol_list[0]))
            contact_info = problem.get_contact_info()
            
            logger.info(f"  Solution time: {solve_time:.2f}s")
            logger.info(f"  Contacts: {contact_info['num_contacts']}")
            logger.info(f"  Total force: {contact_info['total_force']*1e3:.3f} mN")
            
            # Store results
            results['depths'].append(target_depth)
            results['forces'].append(contact_info['total_force'])
            results['num_contacts'].append(contact_info['num_contacts'])
            results['max_penetrations'].append(contact_info['max_penetration'])
            results['contact_areas'].append(contact_info['contact_area'])
            results['mean_pressures'].append(contact_info['mean_pressure'])
            results['max_displacements'].append(max_displacement)
            results['computation_times'].append(solve_time)
            results['max_lambdas'].append(contact_info['max_lambda'])
            
            # Save VTK
            self.save_vtk_step(problem, sol_list[0], step)
            
            # Update for next step
            try:
                plastic_vars = problem.update_int_vars_gp(sol_list[0], problem.internal_vars)
                previous_vars = problem.get_internal_vars_with_lambda()
            except Exception as e:
                logger.warning(f"Failed to update internal variables: {e}")
                previous_vars = problem.get_internal_vars_with_lambda()
            
            previous_solution = sol_list
        
        # Post-processing
        self.save_results(results)
        self.create_figures(results, simulation_params)
        self.print_summary(results, simulation_params)
        
        return results
    
    def load_mesh_and_orientations(self, msh_file, quat_file):
        """Load mesh and crystal orientations"""
        meshio_mesh = meshio.read(msh_file)
        
        if 'hexahedron' in meshio_mesh.cells_dict:
            cells = meshio_mesh.cells_dict['hexahedron']
        else:
            raise ValueError("No hexahedral cells found in mesh")
        
        mesh = Mesh(meshio_mesh.points, cells)
        quat = onp.loadtxt(quat_file)
        
        if 'gmsh:physical' in meshio_mesh.cell_data:
            cell_grain_inds = meshio_mesh.cell_data['gmsh:physical'][0] - 1
        else:
            num_cells = len(cells)
            num_grains = len(quat)
            cell_grain_inds = onp.random.randint(0, num_grains, size=num_cells)
        
        cell_ori_inds = cell_grain_inds
        
        logger.info(f"Mesh loaded: {len(mesh.points)} nodes, {len(cells)} elements")
        logger.info(f"Grains: {len(quat)}")
        
        return mesh, quat, cell_ori_inds
    
    def save_vtk_step(self, problem, displacement, step):
        """Save VTK file for visualization"""
        vtk_file = os.path.join(self.vtk_dir, f'step_{step:04d}.vtu')
        
        cell_infos = []
        point_infos = []
        
        try:
            sigma_data = problem.compute_avg_stress(displacement, problem.internal_vars)
            sigma_von_mises = onp.sqrt(0.5 * (
                (sigma_data[:, 0, 0] - sigma_data[:, 1, 1])**2 +
                (sigma_data[:, 1, 1] - sigma_data[:, 2, 2])**2 +
                (sigma_data[:, 2, 2] - sigma_data[:, 0, 0])**2 +
                6 * (sigma_data[:, 0, 1]**2 + sigma_data[:, 1, 2]**2 + sigma_data[:, 0, 2]**2)
            ))
            
            cell_infos.extend([
                ('sigma_xx', sigma_data[:, 0, 0]),
                ('sigma_yy', sigma_data[:, 1, 1]),
                ('sigma_zz', sigma_data[:, 2, 2]),
                ('von_mises_stress', sigma_von_mises),
                ('grain_id', problem.additional_info[1])
            ])
        except Exception:
            pass
        
        contact_indicator = onp.zeros(len(problem.original_mesh.points))
        lambda_values = onp.zeros(len(problem.original_mesh.points))
        
        if problem.current_contacts:
            for contact in problem.current_contacts:
                node_id = contact['node_id']
                contact_indicator[node_id] = 1.0
                lambda_values[node_id] = problem.lambda_contact.get(node_id, 0.0)
        
        point_infos.extend([
            ('contact_indicator', contact_indicator),
            ('lambda_multiplier', lambda_values)
        ])
        
        save_sol(problem.fes[0], displacement, vtk_file,
                 cell_infos=cell_infos, point_infos=point_infos)
    
    def save_results(self, results):
        """Save numerical results"""
        for key in results:
            results[key] = onp.array(results[key])
        
        for key, data in results.items():
            file_path = os.path.join(self.data_dir, f"{key}.txt")
            onp.savetxt(file_path, data, fmt='%.10e')
        
        npz_file = os.path.join(self.data_dir, "all_results.npz")
        onp.savez(npz_file, **results)
    
    def create_figures(self, results, simulation_params):
        """Create analysis figures"""
        if len(results['depths']) == 0:
            return
        
        # Force-displacement curve
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(results['depths'] * 1e9, results['forces'] * 1e3, 'b-o', linewidth=2)
        ax.set_xlabel('Indentation depth (nm)')
        ax.set_ylabel('Force (mN)')
        ax.set_title(f'Force-Displacement Curve - {simulation_params["indenter_type"].capitalize()}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'force_displacement.png'), dpi=300)
        plt.close()
        
        # Contact evolution
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(results['depths'] * 1e9, results['num_contacts'], 'r-s', linewidth=2)
        ax.set_xlabel('Indentation depth (nm)')
        ax.set_ylabel('Number of contact nodes')
        ax.set_title('Contact Evolution')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'contact_evolution.png'), dpi=300)
        plt.close()
    
    def print_summary(self, results, simulation_params):
        """Print simulation summary"""
        logger.info("Simulation completed")
        logger.info(f"Output directory: {self.output_dir}")
        
        if len(results['forces']) > 0:
            logger.info(f"Maximum force: {onp.max(results['forces'])*1e3:.3f} mN")
            logger.info(f"Maximum contacts: {onp.max(results['num_contacts'])}")
            logger.info(f"Total computation time: {onp.sum(results['computation_times']):.1f} s")
            
            if results['contact_areas'][-1] > 0:
                hardness = results['forces'][-1] / results['contact_areas'][-1] * 1e-9
                logger.info(f"Estimated hardness: {hardness:.2f} GPa")


def main():
    parser = argparse.ArgumentParser(description='Nano-indentation simulation')
    
    parser.add_argument('--num_grains', type=int, default=50)
    parser.add_argument('--domain_size', type=float, nargs=3, default=[5e-6, 5e-6, 2.5e-6])
    parser.add_argument('--mesh_size', choices=['fine', 'medium', 'coarse'], default='medium')
    parser.add_argument('--indenter_type', choices=['berkovich', 'spherical'], default='spherical')
    parser.add_argument('--sphere_radius', type=float, default=1e-6)
    parser.add_argument('--max_depth', type=float, default=2e-7)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--penalty_parameter', type=float, default=1e10)
    parser.add_argument('--contact_tolerance', type=float, default=1e-9)
    parser.add_argument('--max_uzawa_iter', type=int, default=50)
    parser.add_argument('--uzawa_tolerance', type=float, default=1e-6)
    parser.add_argument('--output_dir', type=str, default='output')
    
    args = parser.parse_args()
    
    simulation_params = {
        'num_grains': args.num_grains,
        'domain_size': args.domain_size,
        'mesh_size': args.mesh_size,
        'indenter_type': args.indenter_type,
        'sphere_radius': args.sphere_radius,
        'max_indentation_depth': args.max_depth,
        'num_steps': args.num_steps,
        'penalty_parameter': args.penalty_parameter,
        'contact_tolerance': args.contact_tolerance,
        'max_uzawa_iter': args.max_uzawa_iter,
        'uzawa_tolerance': args.uzawa_tolerance
    }
    
    try:
        simulation = NanoIndentationSimulation(args.output_dir)
        results = simulation.run_simulation(simulation_params)
        return 0
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
