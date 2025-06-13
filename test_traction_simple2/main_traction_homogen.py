import os
import time
import numpy as onp
import jax
import jax.numpy as np
import meshio

from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol

from jax import config
config.update("jax_enable_x64", True)

class HomogeneousIsotropicElasticity(Problem):
    """
    Problème de mécanique des solides pour un matériau élastique isotrope homogène.
    Utilise une formulation en déformations de Green-Lagrange et 2nd tenseur de Piola-Kirchhoff.
    """
    def custom_init(self):
        """
        Initialisation personnalisée du problème élastique.
        Définit le tenseur d'élasticité isotrope C.
        """
        E_modulus = 1.25e5
        nu_poisson = 0.36

        self.C_iso = np.zeros((self.dim, self.dim, self.dim, self.dim))

        C11 = E_modulus * (1. - nu_poisson) / ((1. + nu_poisson) * (1. - 2. * nu_poisson))
        C12 = E_modulus * nu_poisson / ((1. + nu_poisson) * (1. - 2. * nu_poisson))
        C44 = E_modulus / (2. * (1. + nu_poisson))

        self.C_iso = self.C_iso.at[0, 0, 0, 0].set(C11)
        self.C_iso = self.C_iso.at[1, 1, 1, 1].set(C11)
        self.C_iso = self.C_iso.at[2, 2, 2, 2].set(C11)

        self.C_iso = self.C_iso.at[0, 0, 1, 1].set(C12); self.C_iso = self.C_iso.at[1, 1, 0, 0].set(C12)
        self.C_iso = self.C_iso.at[0, 0, 2, 2].set(C12); self.C_iso = self.C_iso.at[2, 2, 0, 0].set(C12)
        self.C_iso = self.C_iso.at[1, 1, 2, 2].set(C12); self.C_iso = self.C_iso.at[2, 2, 1, 1].set(C12)

        self.C_iso = self.C_iso.at[0, 1, 0, 1].set(C44); self.C_iso = self.C_iso.at[1, 0, 1, 0].set(C44)
        self.C_iso = self.C_iso.at[0, 1, 1, 0].set(C44); self.C_iso = self.C_iso.at[1, 0, 0, 1].set(C44)

        self.C_iso = self.C_iso.at[0, 2, 0, 2].set(C44); self.C_iso = self.C_iso.at[2, 0, 2, 0].set(C44)
        self.C_iso = self.C_iso.at[0, 2, 2, 0].set(C44); self.C_iso = self.C_iso.at[2, 0, 0, 2].set(C44)

        self.C_iso = self.C_iso.at[1, 2, 1, 2].set(C44); self.C_iso = self.C_iso.at[2, 1, 2, 1].set(C44)
        self.C_iso = self.C_iso.at[1, 2, 2, 1].set(C44); self.C_iso = self.C_iso.at[2, 1, 1, 2].set(C44)
        
    def get_tensor_map(self):
        """
        Retourne une fonction qui calcule le 1er tenseur de Piola-Kirchhoff (P_pk1)
        à partir du gradient du déplacement (u_grad).
        """
        def get_P_pk1_fn(u_grad_cell_quad, *internal_vars_at_gp_dummy):
            F = u_grad_cell_quad + np.eye(self.dim)
            E_gl = 0.5 * (F.T @ F - np.eye(self.dim))
            S_pk2 = np.einsum('ijkl,kl->ij', self.C_iso, E_gl)
            P_pk1 = F @ S_pk2
            return P_pk1

        return get_P_pk1_fn

    def set_params(self, params):
        """
        Pas de variables internes pour ce modèle élastique simple.
        """
        pass 

    def update_int_vars_gp(self, sol, params):
        """
        Pas de variables internes évolutives pour ce modèle.
        """
        return params 

    def compute_avg_stress(self, sol, params):
        """
        Calcule la contrainte de Cauchy moyenne par cellule.
        """
        u_grads = np.take(sol, self.fes[0].cells, axis=0)[:, None, :, :, None] * \
                  self.fes[0].shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2) 

        get_P_pk1_fn = self.get_tensor_map()
        
        vmap_P_pk1_calc = jax.vmap(jax.vmap(lambda u_grad_val: get_P_pk1_fn(u_grad_val)))
        P_pk1_all = vmap_P_pk1_calc(u_grads)

        F_all = u_grads + np.eye(self.dim)[None, None, :, :]

        def P_to_sigma_cauchy(P, F_mat):
            det_F = np.linalg.det(F_mat)
            return (1. / det_F) * P @ F_mat.T

        vmap_P_to_sigma = jax.vmap(jax.vmap(P_to_sigma_cauchy))
        sigma_cauchy_all = vmap_P_to_sigma(P_pk1_all, F_all) 

        sigma_cell_data = np.sum(sigma_cauchy_all * self.fes[0].JxW[:, :, None, None], axis=1) / \
                          np.sum(self.fes[0].JxW, axis=1)[:, None, None]
        
        return sigma_cell_data


def problem():
    print("[HOMOGENE] Starting homogeneous elastic traction simulation (uniaxial BCs)")
    # 1) Chemins
    case_name  = 'traction_cube' 
    base_dir   = os.path.dirname(__file__)
    if not base_dir: base_dir = "."
    data_dir   = os.path.join(base_dir, 'data')
    
    neper_dir  = os.path.join(data_dir, 'neper', case_name)
    vtk_dir    = os.path.join(data_dir, 'vtk', case_name + '_uniaxial_homogeneous')
    numpy_dir  = os.path.join(data_dir, 'numpy', case_name + '_uniaxial_homogeneous')
    for d in (vtk_dir, numpy_dir): 
        os.makedirs(d, exist_ok=True)
        print(f"[HOMOGENE] Ensured directory: {d}")

    # 2) Lecture du maillage
    print("[HOMOGENE] Reading mesh...")
    msh_path    = os.path.join(neper_dir, "domain.msh")
    if not os.path.exists(msh_path):
        print(f"[HOMOGENE] ERREUR: Fichier de maillage non trouvé à {msh_path}")
        return
    meshio_mesh = meshio.read(msh_path)
    pts         = meshio_mesh.points
    cells       = meshio_mesh.cells_dict['hexahedron']
    print(f"[HOMOGENE] Mesh: points={pts.shape}, cells={cells.shape}")
    mesh = Mesh(pts, cells)

    min_coords = onp.min(pts, axis=0)
    max_coords = onp.max(pts, axis=0)

    # 4) Dimensions du domaine
    Lx_coord_max = float(onp.max(pts[:,0])); Ly_coord_max = float(onp.max(pts[:,1])); Lz_coord_max = float(onp.max(pts[:,2]))
    Lx_coord_min = float(onp.min(pts[:,0])); Ly_coord_min = float(onp.min(pts[:,1])); Lz_coord_min = float(onp.min(pts[:,2]))
    
    actual_Lz = Lz_coord_max - Lz_coord_min
    print(f"[HOMOGENE] Actual length Lz: {actual_Lz}")

    # 5) Déplacements et temps
    target_strain = 0.01
    max_displacement = target_strain * actual_Lz
    disps = onp.linspace(0., max_displacement, 11) 
    ts    = onp.linspace(0., target_strain, 11)
    print(f"[HOMOGENE] Displacement steps: {len(disps)}, Time steps: {len(ts)}")

    # 6) Conditions aux limites
    def bottom_plane_selector(point, index): return np.isclose(point[2], Lz_coord_min, atol=1e-5) 
    def top_plane_selector(point, index): return np.isclose(point[2], Lz_coord_max,  atol=1e-5) 
    def zero_val_fn(point): return 0.0 
    def top_disp_val_fn(d): return lambda point: d

    bottom_nodes_indices = onp.where(onp.isclose(pts[:, 2], Lz_coord_min, atol=1e-5))[0]
    if len(bottom_nodes_indices) == 0: raise ValueError("Aucun nœud trouvé sur la face inférieure.")
    
    origin_target_coords_val = onp.array([Lx_coord_min, Ly_coord_min, Lz_coord_min])
    distances_to_origin_target = onp.linalg.norm(pts[bottom_nodes_indices] - origin_target_coords_val, axis=1)
    min_dist_idx_local_origin = onp.argmin(distances_to_origin_target)
    origin_bottom_node_global_idx = bottom_nodes_indices[min_dist_idx_local_origin]
    origin_node_actual_coords_val = pts[origin_bottom_node_global_idx]
    print(f"[HOMOGENE] Nœud origine pour CLs (ux, uy, uz) = 0 : index {origin_bottom_node_global_idx}, coords {origin_node_actual_coords_val}")

    def create_origin_node_selector(target_coords_closure):
        def origin_node_selector_inner(point, index): 
            return np.logical_and(np.isclose(point[0], target_coords_closure[0], atol=1e-7),
                       np.logical_and(np.isclose(point[1], target_coords_closure[1], atol=1e-7),
                                      np.isclose(point[2], target_coords_closure[2], atol=1e-7)))
        return origin_node_selector_inner
    origin_node_selector_concrete_fn = create_origin_node_selector(origin_node_actual_coords_val)

    dirichlet_bc_info = [
        [bottom_plane_selector, origin_node_selector_concrete_fn, origin_node_selector_concrete_fn, top_plane_selector],
        [2,                     0,                                1,                                2], 
        [zero_val_fn,           zero_val_fn,                      zero_val_fn,                        top_disp_val_fn(disps[0])]
    ]
    print("[HOMOGENE] Dirichlet BC (uniaxial setup) initialized")

    # 7) Initialisation du problème
    print("[HOMOGENE] Initializing HomogeneousIsotropicElasticity problem")
    problem_obj = HomogeneousIsotropicElasticity( 
        mesh,
        vec=3, dim=3, ele_type='HEX8',
        dirichlet_bc_info=dirichlet_bc_info
    )

    # 8) Préparation de la solution et des logs
    sol_prev = onp.zeros((problem_obj.fes[0].num_total_nodes, problem_obj.fes[0].vec)) 
    params = None 
    global_log, local_log, local_2_log, dist_log = [], [], [], []
    cell0 = 0 

    top_node_indices = onp.where(onp.isclose(pts[:, 2], Lz_coord_max, atol=1e-5))[0]
    cell_top_idx = -1
    if len(top_node_indices) > 0:
        top_node_idx_for_cell_search = top_node_indices[0]
        for c_idx, cell_nodes_list in enumerate(cells):
            if top_node_idx_for_cell_search in cell_nodes_list:
                cell_top_idx = c_idx
                break
    if cell_top_idx == -1:
        print("[HOMOGENE] WARNING: Could not find a cell on the top surface. Defaulting to cell 0 for local_2 log.")
        cell_top_idx = 0 
    print(f"[HOMOGENE] Using cell {cell_top_idx} for local_2_homo.txt.")

    # 9) Boucle temporelle
    for i in range(len(ts)-1): 
        print(f"\n[HOMOGENE] Step {i+1}/{len(ts)-1}:")

        disp_val = float(disps[i+1])
        eps_log_val  = disp_val / actual_Lz 
        dt_val   = float(ts[i+1] - ts[i]) 
        problem_obj.dt = dt_val 

        print(f"[HOMOGENE] Updating Dirichlet BC: eps_for_log={eps_log_val:.6f}, disp_applied={disp_val:.6f}")
        current_dirichlet_bc_info = [
            [bottom_plane_selector, origin_node_selector_concrete_fn, origin_node_selector_concrete_fn, top_plane_selector],
            [2,                     0,                                1,                                2], 
            [zero_val_fn,           zero_val_fn,                      zero_val_fn,                        top_disp_val_fn(disp_val)]
        ]
        problem_obj.fes[0].update_Dirichlet_boundary_conditions(current_dirichlet_bc_info) 

        print(f"[HOMOGENE] Solving for eps_for_log={eps_log_val:.6f}...")
        problem_obj.set_params(params) 
        sol_list = solver(problem_obj, {'jax_solver': {}, 'initial_guess': np.array(sol_prev)}) 
        sol_prev = sol_list[0] 

        print(f"[HOMOGENE] Computing stresses...")
        sigma_stresses = problem_obj.compute_avg_stress(sol_prev, params) 
        
        sigma_xx_stress_np = onp.array(sigma_stresses[:,0,0])
        sigma_yy_stress_np = onp.array(sigma_stresses[:,1,1])
        sigma_zz_stress_np = onp.array(sigma_stresses[:,2,2])
        sigma_xy_stress_np = onp.array(sigma_stresses[:,0,1]) 
        sigma_yz_stress_np = onp.array(sigma_stresses[:,1,2]) 
        sigma_xz_stress_np = onp.array(sigma_stresses[:,0,2]) 

        sigma_vm_stress_np = onp.sqrt(
            0.5*((sigma_xx_stress_np - sigma_yy_stress_np)**2 + (sigma_yy_stress_np - sigma_zz_stress_np)**2 + (sigma_zz_stress_np - sigma_xx_stress_np)**2)
          + 3.0*(sigma_xy_stress_np**2 + sigma_yz_stress_np**2 + sigma_xz_stress_np**2)
        )

        print(f"[HOMOGENE] Writing VTU file...")
        vtu_name = f"homo_vtk_uniaxial_step_{i:03d}.vtu"
        vtk_path = os.path.join(vtk_dir, vtu_name)
        save_sol(
            problem_obj.fes[0], sol_prev, vtk_path,
            cell_infos=[ 
                ('sigma_xx',         sigma_xx_stress_np), 
                ('sigma_yy',         sigma_yy_stress_np), 
                ('sigma_zz',         sigma_zz_stress_np), 
                ('von_Mises_stress', sigma_vm_stress_np), 
            ]
        )
        print(f"[HOMOGENE] Wrote VTU: {vtu_name}")

        avg_zz = float(onp.mean(sigma_zz_stress_np)); avg_vm = float(onp.mean(sigma_vm_stress_np)) 
        global_log.append([eps_log_val, avg_zz, avg_vm]) 

        local_z_cell0  = float(sigma_zz_stress_np[cell0]); local_vm_cell0 = float(sigma_vm_stress_np[cell0]) 
        local_log.append([eps_log_val, local_z_cell0, local_vm_cell0]) 

        if cell_top_idx != -1 and cell_top_idx < len(sigma_zz_stress_np):
            local_z_cell_top  = float(sigma_zz_stress_np[cell_top_idx])
            local_vm_cell_top = float(sigma_vm_stress_np[cell_top_idx])
            local_2_log.append([eps_log_val, local_z_cell_top, local_vm_cell_top])
        else: 
            local_2_log.append([eps_log_val, onp.nan, onp.nan]) 

        pcts = onp.percentile(sigma_zz_stress_np, [0,25,50,75,100]) 
        dist_log.append([eps_log_val, *pcts]) 

        params = problem_obj.update_int_vars_gp(sol_prev, params) 

    # 10) Sauvegarde des logs
    onp.savetxt(os.path.join(numpy_dir, 'global_homo.txt'), onp.array(global_log), header='eps sigma_homo_avg sigma_homo_vm', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'local_homo.txt'), onp.array(local_log), header='eps sigma_zz_cell0 sigma_vm_cell0', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'local_2_homo.txt'), onp.array(local_2_log), header='eps sigma_zz_cell_top sigma_vm_cell_top', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'dist_homo.txt'), onp.array(dist_log), header='eps min q25 median q75 max', fmt='%.6e')
    
    print(f"[HOMOGENE] Saved logs to {numpy_dir}")

if __name__ == '__main__':
    start_time = time.time()
    problem()
    print(f"[HOMOGENE] Total time: {time.time() - start_time:.2f}s")
