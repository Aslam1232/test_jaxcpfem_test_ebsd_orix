import numpy as onp
import jax
import jax.numpy as np
import jax.flatten_util
import os
import sys
import functools
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

from jax_fem.problem import Problem

from jax import config
config.update("jax_enable_x64", True)

onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=10)

crt_dir = os.path.dirname(__file__)

class CopperProperties:
    YOUNG_MODULUS = 110e9
    POISSON_RATIO = 0.35
    SHEAR_MODULUS = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))
    
    C11 = YOUNG_MODULUS * (1 - POISSON_RATIO) / ((1 + POISSON_RATIO) * (1 - 2*POISSON_RATIO))
    C12 = YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2*POISSON_RATIO))
    C44 = SHEAR_MODULUS
    
    YIELD_STRESS_INITIAL = 25e6
    
    HARDENING_MODULUS = 200e6
    SATURATION_STRESS = 150e6
    HARDENING_EXPONENT = 2.0
    REFERENCE_STRAIN_RATE = 0.001
    RATE_SENSITIVITY = 1.0/20.0
    
    LATENT_HARDENING_RATIO = 1.4
    
    @classmethod
    def validate_properties(cls):
        issues = []
        if cls.YOUNG_MODULUS <= 0: issues.append("Module de Young négatif ou nul")
        if not (0 < cls.POISSON_RATIO < 0.5): issues.append("Coefficient de Poisson hors limites physiques")
        if cls.C11 <= cls.C12: issues.append("Condition de stabilité C11 > C12 violée")
        if cls.C44 <= 0: issues.append("Module de cisaillement négatif")
        if cls.YIELD_STRESS_INITIAL >= cls.SATURATION_STRESS: issues.append("Contrainte initiale >= contrainte de saturation")
        return issues

@lru_cache(maxsize=128)
def rotate_tensor_rank_4_cached(R_hash, T_hash):
    pass

def rotate_tensor_rank_4(R, T):
    R0 = R[:, :, None, None, None, None, None, None]
    R1 = R[None, None, :, :, None, None, None, None]
    R2 = R[None, None, None, None, :, :, None, None]
    R3 = R[None, None, None, None, None, None, :, :]
    result = np.sum(R0 * R1 * R2 * R3 * T[None, :, None, :, None, :, None, :], axis=(1, 3, 5, 7))
    return result

def rotate_tensor_rank_2(R, T):
    R0 = R[:, :, None, None]
    R1 = R[None, None, :, :]
    return np.sum(R0 * R1 * T[None, :, None, :], axis=(1, 3))

rotate_tensor_rank_2_vmap = jax.jit(jax.vmap(rotate_tensor_rank_2, in_axes=(None, 0)))
rotate_tensor_rank_4_vmap = jax.jit(jax.vmap(rotate_tensor_rank_4, in_axes=(0, None)))

def get_rot_mat(q):
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        raise ValueError(f"Quaternion trop petit: norme = {q_norm}")
    q = q / q_norm
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return np.array([
        [q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3]
    ])

get_rot_mat_vmap = jax.jit(jax.vmap(get_rot_mat))

class FCCSlipSystems:
    def __init__(self):
        self._slip_systems = None
        self._schmid_tensors = None
        self._interaction_matrix = None
    
    @lru_cache(maxsize=1)
    def get_slip_systems(self):
        if self._slip_systems is not None:
            return self._slip_systems
        slip_planes = onp.array([[ 1,  1,  1], [-1,  1,  1], [ 1, -1,  1], [ 1,  1, -1]], dtype=onp.float64)
        slip_directions_base = onp.array([[ 1, -1,  0], [-1,  0,  1], [ 0,  1, -1]], dtype=onp.float64)
        slip_systems = []
        for i, plane_normal in enumerate(slip_planes):
            plane_normal = plane_normal / onp.linalg.norm(plane_normal)
            for j, base_direction in enumerate(slip_directions_base):
                direction = base_direction - onp.dot(base_direction, plane_normal) * plane_normal
                direction_norm = onp.linalg.norm(direction)
                if direction_norm < 1e-12: continue
                direction = direction / direction_norm
                dot_product = onp.abs(onp.dot(plane_normal, direction))
                if dot_product > 1e-10: raise ValueError(f"Plan et direction non orthogonaux: dot = {dot_product}")
                slip_systems.append(onp.concatenate([plane_normal, direction]))
        self._slip_systems = onp.array(slip_systems)
        if len(self._slip_systems) != 12:
            raise ValueError(f"Nombre incorrect de systèmes de glissement: {len(self._slip_systems)}")
        return self._slip_systems
    
    @lru_cache(maxsize=1)
    def get_schmid_tensors(self):
        if self._schmid_tensors is not None: return self._schmid_tensors
        slip_systems = self.get_slip_systems()
        slip_normals = slip_systems[:, :3]
        slip_directions = slip_systems[:, 3:]
        self._schmid_tensors = jax.vmap(np.outer)(slip_directions, slip_normals)
        return self._schmid_tensors
    
    @lru_cache(maxsize=1)
    def get_interaction_matrix(self, latent_ratio=1.4):
        if self._interaction_matrix is not None: return self._interaction_matrix
        slip_systems = self.get_slip_systems()
        num_systems = len(slip_systems)
        interaction_matrix = latent_ratio * onp.ones((num_systems, num_systems))
        onp.fill_diagonal(interaction_matrix, 1.0)
        slip_normals = slip_systems[:, :3]
        for i in range(num_systems):
            for j in range(i+1, num_systems):
                if onp.abs(onp.dot(slip_normals[i], slip_normals[j])) > 0.99:
                    interaction_matrix[i, j] = 1.0
                    interaction_matrix[j, i] = 1.0
        self._interaction_matrix = interaction_matrix
        if not onp.allclose(interaction_matrix, interaction_matrix.T): raise ValueError("Matrice d'interaction non symétrique")
        return self._interaction_matrix

class CrystalPlasticityCuivre(Problem):
    def custom_init(self, quat, cell_ori_inds):
        self.additional_info = (quat, cell_ori_inds) 
        property_issues = CopperProperties.validate_properties()
        if property_issues:
            raise ValueError(f"Propriétés matériau invalides: {property_issues}")

        self.fcc_systems = FCCSlipSystems()
        
        try:
            input_slip_sys = self.fcc_systems.get_slip_systems()
            num_slip_sys = len(input_slip_sys)
            self.Schmid_tensors = self.fcc_systems.get_schmid_tensors()
            self.q = self.fcc_systems.get_interaction_matrix(CopperProperties.LATENT_HARDENING_RATIO)
            rot_mats = onp.array(get_rot_mat_vmap(quat)[cell_ori_inds])
            
            num_cells = len(self.fes[0].cells)
            num_quads = self.fes[0].num_quads
            
            Fp_inv_gp = onp.repeat(onp.eye(self.dim)[None, None, :, :], num_cells * num_quads).reshape(num_cells, num_quads, self.dim, self.dim)
            slip_resistance_gp = CopperProperties.YIELD_STRESS_INITIAL * onp.ones((num_cells, num_quads, num_slip_sys))
            slip_gp = onp.zeros_like(slip_resistance_gp)
            rot_mats_gp = onp.repeat(rot_mats[:, None, :, :], num_quads, axis=1)
            
            self.C = onp.zeros((self.dim, self.dim, self.dim, self.dim))
            C11, C12, C44 = CopperProperties.C11, CopperProperties.C12, CopperProperties.C44
            self.C[0, 0, 0, 0], self.C[1, 1, 1, 1], self.C[2, 2, 2, 2] = C11, C11, C11
            self.C[0, 0, 1, 1], self.C[1, 1, 0, 0] = C12, C12
            self.C[0, 0, 2, 2], self.C[2, 2, 0, 0] = C12, C12
            self.C[1, 1, 2, 2], self.C[2, 2, 1, 1] = C12, C12
            shear_indices = [(1, 2, 1, 2), (1, 2, 2, 1), (2, 1, 1, 2), (2, 1, 2, 1),
                             (2, 0, 2, 0), (2, 0, 0, 2), (0, 2, 2, 0), (0, 2, 0, 2),
                             (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]
            for idx in shear_indices: self.C[idx] = C44

            self.internal_vars = [Fp_inv_gp, slip_resistance_gp, slip_gp, rot_mats_gp]
            self.dt = 0.01
            self._cached_maps = None
            self._maps_cache_valid = False
            self.get_kernel()
            
        except Exception as e:
            print(f"[ERROR] Erreur dans custom_init: {e}")
            raise

    def set_adaptive_timestep(self, solution_increment=None):
        try:
            if solution_increment is not None:
                max_strain_inc = onp.max(onp.abs(solution_increment))
                if max_strain_inc > 0:
                    deformation_rate = max_strain_inc / self.dt
                    if deformation_rate > 0.1: self.dt = min(self.dt, 0.005)
                    elif deformation_rate < 0.01: self.dt = min(self.dt * 1.1, 0.02)
            self.dt = onp.clip(self.dt, 1e-4, 0.05)
        except Exception as e:
            self.dt = 0.01
    
    def get_maps(self):
        if self._cached_maps is not None and self._maps_cache_valid:
            return self._cached_maps
        
        h = CopperProperties.HARDENING_MODULUS
        t_sat = CopperProperties.SATURATION_STRESS
        gss_a = CopperProperties.HARDENING_EXPONENT
        ao = CopperProperties.REFERENCE_STRAIN_RATE
        xm = CopperProperties.RATE_SENSITIVITY
        
        def get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat):
            _, unflatten_fn = jax.flatten_util.ravel_pytree(Fp_inv_old)
            _, unflatten_fn_params = jax.flatten_util.ravel_pytree([
                Fp_inv_old, Fp_inv_old, slip_resistance_old, slip_old, rot_mat
            ])
            
            def first_PK_stress(u_grad):
                x, _ = jax.flatten_util.ravel_pytree([u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat])
                y = newton_solver(x)
                S = unflatten_fn(y)
                _, _, _, Fe, F = helper(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, S)
                Fe_det = np.linalg.det(Fe)
                sigma = (1.0 / np.maximum(Fe_det, 1e-12)) * Fe @ S @ Fe.T
                F_det = np.linalg.det(F)
                F_inv_T = np.linalg.inv(F).T
                P = np.maximum(F_det, 1e-12) * sigma @ F_inv_T
                return P
            
            def update_int_vars(u_grad):
                x, _ = jax.flatten_util.ravel_pytree([u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat])
                y = newton_solver(x)
                S = unflatten_fn(y)
                Fp_inv_new, slip_resistance_new, slip_new, _, _ = helper(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, S)
                return Fp_inv_new, slip_resistance_new, slip_new, rot_mat
            
            def helper(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, S):
                tau = np.sum(S[None, :, :] * rotate_tensor_rank_2_vmap(rot_mat, self.Schmid_tensors), axis=(1, 2))
                safe_resistance = np.maximum(slip_resistance_old, 1e-6)
                gamma_inc = ao * self.dt * (np.abs(tau / safe_resistance))**(1.0 / xm) * np.sign(tau)
                saturation_ratio = np.minimum(slip_resistance_old / t_sat, 0.999)
                hardening_term = np.abs(1.0 - saturation_ratio)**gss_a * np.sign(1.0 - saturation_ratio)
                g_inc_local = h * np.abs(gamma_inc) * hardening_term
                g_inc = (self.q @ g_inc_local[:, None]).reshape(-1)
                slip_resistance_new = slip_resistance_old + g_inc
                slip_new = slip_old + gamma_inc
                F = u_grad + np.eye(self.dim)
                L_plastic_inc = np.sum(gamma_inc[:, None, None] * rotate_tensor_rank_2_vmap(rot_mat, self.Schmid_tensors), axis=0)
                Fp_inv_new = Fp_inv_old @ (np.eye(self.dim) - L_plastic_inc)
                Fe = F @ Fp_inv_new
                return Fp_inv_new, slip_resistance_new, slip_new, Fe, F
            
            def implicit_residual(x, y):
                u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat = unflatten_fn_params(x)
                S = unflatten_fn(y)
                _, _, _, Fe, _ = helper(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat, S)
                E_elastic = 0.5 * (Fe.T @ Fe - np.eye(self.dim))
                S_constitutive = np.sum(rotate_tensor_rank_4(rot_mat, self.C) * E_elastic[None, None, :, :], axis=(2, 3))
                res, _ = jax.flatten_util.ravel_pytree(S - S_constitutive)
                return res
            
            @jax.custom_jvp
            def newton_solver(x):
                y0 = np.zeros(self.dim * self.dim)
                max_steps, tol = 20, 1e-8
                res_vec = implicit_residual(x, y0)
                def cond_fun(state):
                    step, res_vec, y = state
                    return np.logical_and(np.linalg.norm(res_vec) > tol, step < max_steps)
                def body_fun(state):
                    step, res_vec, y = state
                    f_partial = lambda y_var: implicit_residual(x, y_var)
                    jac = jax.jacfwd(f_partial)(y)
                    jac = jac + 1e-8 * np.eye(len(jac))
                    y_inc = np.linalg.solve(jac, -res_vec)
                    return step + 1, f_partial(y + y_inc), y + y_inc
                _, res_vec_f, y_f = jax.lax.while_loop(cond_fun, body_fun, (0, res_vec, y0))
                return y_f
            
            @newton_solver.defjvp
            def f_jvp(primals, tangents):
                x, = primals; v, = tangents
                y = newton_solver(x)
                jac_x = jax.jacfwd(implicit_residual, argnums=0)(x, y)
                jac_y = jax.jacfwd(implicit_residual, argnums=1)(x, y)
                jac_y = jac_y + 1e-8 * np.eye(len(jac_y))
                jvp_result = np.linalg.solve(jac_y, -(jac_x @ v[:, None]).reshape(-1))
                return y, jvp_result
            
            return first_PK_stress, update_int_vars
        
        def tensor_map(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat):
            first_PK_stress, _ = get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat)
            return first_PK_stress(u_grad)
        
        def update_int_vars_map(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat):
            _, update_int_vars = get_partial_tensor_map(Fp_inv_old, slip_resistance_old, slip_old, rot_mat)
            return update_int_vars(u_grad)
        
        self._cached_maps = (tensor_map, update_int_vars_map)
        self._maps_cache_valid = True
        return self._cached_maps
    
    def get_kernel(self):
        tensor_map, _ = self.get_maps()
        self.kernel = lambda u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat: \
                      tensor_map(u_grad, Fp_inv_old, slip_resistance_old, slip_old, rot_mat)
    
    def get_tensor_map(self):
        return self.get_maps()[0]
    
    def update_int_vars_gp(self, sol, params):
        _, update_int_vars_map = self.get_maps()
        vmap_update_int_vars_map = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
        u_grads = np.sum(np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :], axis=2)
        self.set_adaptive_timestep(solution_increment=sol)
        try:
            return vmap_update_int_vars_map(u_grads, *params)
        except Exception as e:
            print(f"[ERROR] Mise à jour variables échouée: {e}")
            return params
    
    def set_params(self, params):
        self.internal_vars = params
        self._maps_cache_valid = False
    
    def compute_avg_stress(self, sol, params):
        u_grads = np.sum(np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :], axis=2)
        tensor_map, _ = self.get_maps()
        vmap_tensor_map = jax.jit(jax.vmap(jax.vmap(tensor_map)))
        P = vmap_tensor_map(u_grads, *params)
        def P_to_sigma(P, F):
            F_det = np.linalg.det(F)
            return (1.0 / np.maximum(F_det, 1e-12)) * P @ F.T
        vvmap_P_to_sigma = jax.vmap(jax.vmap(P_to_sigma))
        F = u_grads + np.eye(self.dim)[None, None, :, :]
        sigma = vvmap_P_to_sigma(P, F)
        sigma_cell_data = np.sum(sigma * self.fes[0].JxW[:, :, None, None], 1) / np.sum(self.fes[0].JxW, axis=1)[:, None, None]
        return sigma_cell_data

CrystalPlasticity = CrystalPlasticityCuivre

__all__ = [
    'CrystalPlasticityCuivre',
    'CrystalPlasticity',
    'CopperProperties',
    'FCCSlipSystems', 
    'get_rot_mat_vmap'
]
