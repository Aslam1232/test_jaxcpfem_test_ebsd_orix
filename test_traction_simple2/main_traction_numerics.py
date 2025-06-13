import os
import time
import numpy as onp
import jax.numpy as np
import meshio

from jax_fem.generate_mesh import Mesh
from jax_fem.solver       import solver
from jax_fem.utils        import save_sol
from applications.polycrystal_304steel.models_304steel import CrystalPlasticity


def problem():
    print("[POLY] Starting polycrystalline traction simulation (uniaxial BCs)")
    # 1) Définition des chemins
    case_name  = 'traction_cube'
    base_dir   = os.path.dirname(__file__)
    if not base_dir:
        base_dir = "."
    data_dir   = os.path.join(base_dir, 'data')
    
    neper_dir  = os.path.join(data_dir, 'neper', case_name)
    csv_dir    = os.path.join(data_dir, 'csv',   case_name)
    vtk_dir    = os.path.join(data_dir, 'vtk',   case_name + '_uniaxial_debug')
    numpy_dir  = os.path.join(data_dir, 'numpy', case_name + '_uniaxial_debug')
    for d in (neper_dir, csv_dir, vtk_dir, numpy_dir):
        os.makedirs(d, exist_ok=True)
        print(f"[POLY] Ensured directory: {d}")

    # 2) Lecture du maillage
    print("[POLY] Reading mesh...")
    msh_path    = os.path.join(neper_dir, "domain.msh")
    if not os.path.exists(msh_path):
        print(f"[POLY] ERREUR: Fichier de maillage non trouvé à {msh_path}")
        return
    meshio_mesh = meshio.read(msh_path)
    pts         = meshio_mesh.points
    cells       = meshio_mesh.cells_dict['hexahedron']
    if isinstance(meshio_mesh.cell_data['gmsh:physical'], list):
        grain_tags = meshio_mesh.cell_data['gmsh:physical'][0]
    else:
        grain_tags = meshio_mesh.cell_data['gmsh:physical']
    print(f"[POLY] Mesh: points={pts.shape}, cells={cells.shape}")
    cell_grain_inds = grain_tags - 1 
    mesh = Mesh(pts, cells)

    # 3) Chargement des quaternions
    print("[POLY] Loading quaternions...")
    quat_file_path = os.path.join(csv_dir, "quat.txt")
    if not os.path.exists(quat_file_path):
        print(f"[POLY] ERREUR: Fichier de quaternions non trouvé à {quat_file_path}")
        return
    quat_full = onp.loadtxt(quat_file_path)
    if quat_full.ndim == 1:
        quat_full = quat_full.reshape(1, -1)

    quat      = quat_full[:, 1:] 
    grain_oris_inds = onp.arange(quat.shape[0]) 
    cell_ori_inds   = onp.take(grain_oris_inds, cell_grain_inds, axis=0) 
    print(f"[POLY] Number of grains from quat.txt: {quat.shape[0]}")

    # 4) Dimensions du domaine
    Lx_coord_max = float(onp.max(pts[:,0])); Ly_coord_max = float(onp.max(pts[:,1])); Lz_coord_max = float(onp.max(pts[:,2]))
    Lx_coord_min = float(onp.min(pts[:,0])); Ly_coord_min = float(onp.min(pts[:,1])); Lz_coord_min = float(onp.min(pts[:,2]))
    
    actual_Lx = Lx_coord_max - Lx_coord_min
    actual_Ly = Ly_coord_max - Ly_coord_min
    actual_Lz = Lz_coord_max - Lz_coord_min

    print(f"[POLY] Max coords: Lx_coord_max={Lx_coord_max}, Ly_coord_max={Ly_coord_max}, Lz_coord_max={Lz_coord_max}")
    print(f"[POLY] Actual lengths: actual_Lx={actual_Lx}, actual_Ly={actual_Ly}, actual_Lz={actual_Lz}")

    # 5) Définition des déplacements et du temps
    target_strain = 0.01
    max_displacement = target_strain * actual_Lz
    disps = onp.linspace(0., max_displacement, 11) 
    ts    = onp.linspace(0., target_strain, 11)

    print(f"[POLY] Displacement steps: {len(disps)}, Time steps: {len(ts)}")

    # 6) Fonctions pour les conditions aux limites
    def bottom_plane_selector(point, index):   
        return np.isclose(point[2], Lz_coord_min, atol=1e-5) 
        
    def top_plane_selector(point, index):      
        return np.isclose(point[2], Lz_coord_max,  atol=1e-5) 
        
    def zero_val_fn(point): 
        return 0.0 
        
    def top_disp_val_fn(d): 
        def val_lambda(point): 
            return d
        return val_lambda

    bottom_nodes_indices = onp.where(onp.isclose(pts[:, 2], Lz_coord_min, atol=1e-5))[0]
    if len(bottom_nodes_indices) == 0:
        raise ValueError("Aucun nœud trouvé sur la face inférieure.")
    
    bottom_pts_coords = pts[bottom_nodes_indices]
    origin_target_coords_val = onp.array([Lx_coord_min, Ly_coord_min, Lz_coord_min])
    
    distances_to_origin_target = onp.linalg.norm(bottom_pts_coords - origin_target_coords_val, axis=1)
    min_dist_idx_local_origin = onp.argmin(distances_to_origin_target)
    origin_bottom_node_global_idx = bottom_nodes_indices[min_dist_idx_local_origin]
    origin_node_actual_coords_val = pts[origin_bottom_node_global_idx]
    
    print(f"[POLY] Nœud sur face inférieure le plus proche de {origin_target_coords_val} (pour CLs ux, uy, uz) : index {origin_bottom_node_global_idx}, coords {origin_node_actual_coords_val}")

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
    print("[POLY] Dirichlet BC (uniaxial setup) initialized")

    # 7) Initialisation du problème
    print("[POLY] Initializing CrystalPlasticity problem")
    problem_obj = CrystalPlasticity( 
        mesh,
        vec=3, dim=3, ele_type='HEX8',
        dirichlet_bc_info=dirichlet_bc_info,
        additional_info=(quat, cell_ori_inds)
    )

    # 8) Préparation de la solution et des logs
    sol_prev    = onp.zeros((problem_obj.fes[0].num_total_nodes, problem_obj.fes[0].vec)) 
    params      = problem_obj.internal_vars 
    global_log, local_log, local_2_log, dist_log, slip_log = [], [], [], [], []
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
        print("[POLY] WARNING: Could not find a cell on the top surface. Defaulting to cell 0.")
        cell_top_idx = 0 
    print(f"[POLY] Using cell {cell_top_idx} for local_2_poly.txt.")

    # 9) Boucle temporelle
    for i in range(len(ts)-1): 
        print(f"\n[POLY] Step {i+1}/{len(ts)-1}:")

        disp_val = float(disps[i+1])
        eps_log_val  = disp_val / actual_Lz 
        dt_val   = float(ts[i+1] - ts[i]) 
        problem_obj.dt = dt_val 

        print(f"[POLY] Updating Dirichlet BC: eps_for_log={eps_log_val:.6f}, disp_applied={disp_val:.6f}")
        current_dirichlet_bc_info = [
            [bottom_plane_selector, origin_node_selector_concrete_fn, origin_node_selector_concrete_fn, top_plane_selector],
            [2,                     0,                                1,                                2], 
            [zero_val_fn,           zero_val_fn,                      zero_val_fn,                        top_disp_val_fn(disp_val)]
        ]
        problem_obj.fes[0].update_Dirichlet_boundary_conditions(current_dirichlet_bc_info) 

        print(f"[POLY] Solving for eps_for_log={eps_log_val:.6f}...")
        problem_obj.set_params(params) 
        sol_list = solver(problem_obj, {'jax_solver': {}, 'initial_guess': np.array(sol_prev)}) 
        sol_prev = sol_list[0] 

        print(f"[POLY] Computing stresses...")
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

        print(f"[POLY] Writing VTU file...")
        vtu_name = f"poly_vtk_uniaxial_step_{i:03d}.vtu"
        vtk_path = os.path.join(vtk_dir, vtu_name)
        save_sol(
            problem_obj.fes[0], sol_prev, vtk_path,
            cell_infos=[
                ('cell_ori_inds',    cell_ori_inds.astype(onp.float64) if cell_ori_inds is not None else None),
                ('sigma_xx',         sigma_xx_stress_np), 
                ('sigma_yy',         sigma_yy_stress_np), 
                ('sigma_zz',         sigma_zz_stress_np), 
                ('von_Mises_stress', sigma_vm_stress_np), 
            ]
        )
        print(f"[POLY] Wrote VTU: {vtu_name}")

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
        
        if isinstance(params[2], (list, tuple)):
             slip_values_for_max = onp.array(params[2])
        else:
             slip_values_for_max = params[2]
        slip_max = float(onp.max(onp.abs(onp.array(slip_values_for_max))))
        slip_log.append([eps_log_val, slip_max]) 

    # 10) Sauvegarde des logs
    onp.savetxt(os.path.join(numpy_dir, 'global_poly.txt'), onp.array(global_log), header='eps sigma_poly_avg sigma_poly_vm', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'local_poly.txt'), onp.array(local_log), header='eps sigma_zz_cell0 sigma_vm_cell0', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'local_2_poly.txt'), onp.array(local_2_log), header='eps sigma_zz_cell_top sigma_vm_cell_top', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'dist_poly.txt'), onp.array(dist_log), header='eps min q25 median q75 max', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'slip_poly.txt'), onp.array(slip_log), header='eps max_slip', fmt='%.6e')
    
    print(f"[POLY] Saved logs to {numpy_dir}")

if __name__ == '__main__':
    start = time.time()
    problem()
    print(f"[POLY] Total time: {time.time() - start:.2f}s")
