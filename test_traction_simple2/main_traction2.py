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
    print("[POLY] Starting polycrystalline traction simulation (debug mode)")
    
    # 1) Définition des chemins
    case_name  = 'traction_cube'
    base_dir   = os.path.dirname(__file__)
    data_dir   = os.path.join(base_dir, 'data')
    neper_dir  = os.path.join(data_dir, 'neper', case_name)
    csv_dir    = os.path.join(data_dir, 'csv',   case_name)
    vtk_dir    = os.path.join(data_dir, 'vtk',   case_name + '_debug')
    numpy_dir  = os.path.join(data_dir, 'numpy', case_name + '_debug')
    for d in (neper_dir, csv_dir, vtk_dir, numpy_dir):
        os.makedirs(d, exist_ok=True)
        print(f"[POLY] Ensured directory: {d}")

    # 2) Lecture du maillage
    print("[POLY] Reading mesh...")
    msh_path    = os.path.join(neper_dir, "domain.msh")
    meshio_mesh = meshio.read(msh_path)
    pts         = meshio_mesh.points
    cells       = meshio_mesh.cells_dict['hexahedron']
    grain_tags  = meshio_mesh.cell_data['gmsh:physical'][0]
    print(f"[POLY] Mesh: points={pts.shape}, cells={cells.shape}")
    cell_grain_inds = grain_tags - 1
    mesh = Mesh(pts, cells)

    # 3) Chargement des quaternions
    print("[POLY] Loading quaternions...")
    quat_full = onp.loadtxt(os.path.join(csv_dir, "quat.txt"))
    quat      = quat_full[:, 1:]
    grain_oris_inds = onp.arange(quat.shape[0])
    cell_ori_inds   = onp.take(grain_oris_inds, cell_grain_inds, axis=0)
    print(f"[POLY] Number of grains: {quat.shape[0]}")

    # 4) Dimensions du domaine
    Lx = float(np.max(pts[:,0])); Ly = float(np.max(pts[:,1])); Lz = float(np.max(pts[:,2]))
    print(f"[POLY] Domain size: Lx={Lx}, Ly={Ly}, Lz={Lz}")

    # 5) Définition des déplacements et du temps
    disps = np.linspace(0., 0.01 * Lx, 11)
    ts    = np.linspace(0., 0.01,        11)
    print(f"[POLY] Displacement steps: {len(disps)}, Time steps: {len(ts)}")

    # 6) Fonctions pour les conditions aux limites
    def bottom(point):   return np.isclose(point[2], 0.0, atol=1e-5)
    def top(point):      return np.isclose(point[2], Lz,  atol=1e-5)
    def zero_val(point): return 0.0
    def top_disp_val(d): return lambda pt: d

    dirichlet_bc_info = [
        [bottom,    bottom,    bottom,   top],
        [0,         1,         2,        2],
        [zero_val,  zero_val,  zero_val, top_disp_val(disps[0])]
    ]
    print("[POLY] Dirichlet BC initialized")

    # 7) Initialisation du problème de plasticité cristalline
    print("[POLY] Initializing CrystalPlasticity problem")
    problem = CrystalPlasticity(
        mesh,
        vec=3, dim=3, ele_type='HEX8',
        dirichlet_bc_info=dirichlet_bc_info,
        additional_info=(quat, cell_ori_inds)
    )

    # 8) Préparation de la solution et des logs
    sol_prev    = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))
    params      = problem.internal_vars
    global_log  = []
    local_log   = []
    dist_log    = []
    slip_log    = []
    cell0 = 0

    # 9) Boucle temporelle
    for i in range(len(ts)-1):
        print(f"\n[POLY] Step {i+1}/{len(ts)-1}:")

        disp = float(disps[i+1])
        eps  = disp / Lz
        dt   = float(ts[i+1] - ts[i])
        problem.dt = dt

        # Mise à jour des conditions aux limites
        print(f"[POLY] Updating Dirichlet BC: eps={eps:.6f}, disp={disp:.6f}")
        dirichlet_bc_info[-1][-1] = top_disp_val(disp)
        problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)

        # Résolution
        print(f"[POLY] Solving for eps={eps:.6f}...")   
        problem.set_params(params)
        sol_list = solver(problem, {'jax_solver': {}, 'initial_guess': sol_prev})
        sol_prev = sol_list[0]

        # Calcul des contraintes
        print(f"[POLY] Computing stresses...")
        sigma = problem.compute_avg_stress(sol_prev, params)
        sigma_xx = sigma[:,0,0]
        sigma_yy = sigma[:,1,1]
        sigma_zz = sigma[:,2,2]
        sigma_vm = (
            0.5*((sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2)
          + 3.0*(sigma[:,0,1]**2 + sigma[:,1,2]**2 + sigma[:,0,2]**2)
        )**0.5

        # Écriture du fichier VTU
        print(f"[POLY] Writing VTU file...")
        vtu_name = f"poly_test3_vtk_step_{i:03d}.vtu"
        vtk_path = os.path.join(vtk_dir, vtu_name)
        save_sol(
            problem.fes[0], sol_prev, vtk_path,
            cell_infos=[
                ('cell_ori_inds',    cell_ori_inds),
                ('sigma_xx',         sigma_xx),
                ('sigma_yy',         sigma_yy),
                ('sigma_zz',         sigma_zz),
                ('von_Mises_stress', sigma_vm),
            ]
        )
        print(f"[POLY] Wrote VTU: {vtu_name}")

        # Enregistrement des logs
        avg_zz = float(onp.mean(sigma_zz)); avg_vm = float(onp.mean(sigma_vm))
        global_log.append([eps, avg_zz, avg_vm])

        local_z  = float(sigma_zz[cell0]); local_vm = float(sigma_vm[cell0])
        local_log.append([eps, local_z, local_vm])

        pcts = onp.percentile(sigma_zz, [0,25,50,75,100])
        dist_log.append([eps, *pcts])

        # Mise à jour des variables internes et enregistrement du glissement
        params = problem.update_int_vars_gp(sol_prev, params)
        slip_max = float(onp.max(onp.abs(params[2])))
        slip_log.append([eps, slip_max])

    # 10) Sauvegarde des logs dans des fichiers texte
    onp.savetxt(
        os.path.join(numpy_dir, 'global_poly.txt'),  onp.array(global_log),
        header='eps sigma_poly_avg sigma_poly_vm', fmt='%.6e'
    )
    onp.savetxt(
        os.path.join(numpy_dir, 'local_poly.txt'),   onp.array(local_log),
        header='eps sigma_zz_cell0 sigma_vm_cell0', fmt='%.6e'
    )
    onp.savetxt(
        os.path.join(numpy_dir, 'dist_poly.txt'),    onp.array(dist_log),
        header='eps min q25 median q75 max', fmt='%.6e'
    )
    onp.savetxt(
        os.path.join(numpy_dir, 'slip_poly.txt'),    onp.array(slip_log),
        header='eps max_slip', fmt='%.6e'
    )
    print(f"[POLY] Saved logs to {numpy_dir}")

if __name__ == '__main__':
    start = time.time()
    problem()
    print(f"[POLY] Total time: {time.time() - start:.2f}s")
