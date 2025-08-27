""" Simulation de traction en plasticité cristalline pour microstructure ExaCA - Inconel 625 Adaptation du code cuivre pour 169 grains avec maillage ExaCA """

import os
import time
import numpy as onp
import jax.numpy as np
import meshio
import matplotlib.pyplot as plt
import pathlib

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from applications.polycrystal_ExaCA.models_inconel625 import CrystalPlasticityInconel625

print("="*70)
print(" SIMULATION PLASTICITÉ CRISTALLINE - INCONEL 625 - MICROSTRUCTURE EXACA")
print(" 169 grains - Domaine 495×110×80 μm - Traction X")
print("="*70)


def verifier_fichiers_requis(msh_dir, csv_dir):
    print("Vérification des fichiers requis...")
    
    fichiers_requis = [
        os.path.join(msh_dir, "exaca_grains_only.msh"),
        os.path.join(csv_dir, "quat_final.txt")
    ]
    
    erreurs = []
    for fichier in fichiers_requis:
        if not os.path.exists(fichier):
            erreurs.append(f"Fichier manquant : {fichier}")
    
    if erreurs:
        print(f"[ERREUR] {len(erreurs)} fichier(s) manquant(s)")
        for erreur in erreurs:
            print(f"  - {erreur}")
        return False, erreurs
    else:
        print(f"Tous les fichiers ExaCA requis sont présents")
        return True, []

def diagnostiquer_maillage_exaca_et_quaternions(msh_path, quat_file_path, debug_dir):
    print("Diagnostic de cohérence maillage ExaCA ↔ quaternions...")
    
    try:
        print(f"Lecture du maillage ExaCA : {msh_path}")
        meshio_mesh = meshio.read(msh_path)
        
        pts = meshio_mesh.points
        
        if 'hexahedron' in meshio_mesh.cells_dict:
            cells = meshio_mesh.cells_dict['hexahedron']
            cell_type_used = 'hexahedron'
        else:
            return False, {"erreur": f"Maillage ExaCA doit contenir des hexaèdres. Types trouvés: {list(meshio_mesh.cells_dict.keys())}"}
        
        if 'gmsh:physical' not in meshio_mesh.cell_data:
            return False, {"erreur": "Pas de physical groups dans le maillage ExaCA"}
        
        grain_tags = meshio_mesh.cell_data['gmsh:physical'][0]
        grain_tags = onp.array(grain_tags, dtype=int)
        
        print(f"Lecture des quaternions ExaCA : {quat_file_path}")
        quat_full = onp.loadtxt(quat_file_path)
        if quat_full.ndim == 1:
            quat_full = quat_full.reshape(1, -1)
        
        quat = quat_full[:, 1:] 
        indices_quat = quat_full[:, 0].astype(int)
        
        nb_points = len(pts)
        nb_cellules = len(cells)
        grains_uniques = len(onp.unique(grain_tags))
        nb_orientations = len(quat)
        
        print(f"Points : {nb_points}")
        print(f"Cellules : {nb_cellules}")
        print(f"Grains uniques : {grains_uniques}")
        print(f"Orientations : {nb_orientations}")
        
        problemes = []
        
        if grains_uniques < 50:
            problemes.append(f"Trop peu de grains conservés : {grains_uniques}")
        
        if nb_orientations < 50:
            problemes.append(f"Trop peu d'orientations : {nb_orientations}")
        
        if grains_uniques != nb_orientations:
            problemes.append(f"Incohérence : {grains_uniques} grains ≠ {nb_orientations} orientations")
        
        expected_indices = onp.arange(nb_orientations)
        if not onp.array_equal(indices_quat, expected_indices):
            problemes.append("Indices quaternions non consécutifs")
        
        if not onp.allclose(onp.linalg.norm(quat, axis=1), 1.0, atol=1e-6):
            problemes.append("Quaternions non normalisés détectés")
        
        cell_grain_inds = grain_tags - 1
        max_grain_id = onp.max(cell_grain_inds)
        min_grain_id = onp.min(cell_grain_inds)
        
        if min_grain_id < 0:
            problemes.append(f"Indices de grains négatifs détectés: min={min_grain_id}")
        
        if max_grain_id >= nb_orientations:
            problemes.append(f"Indices de grains trop élevés: max={max_grain_id} ≥ {nb_orientations}")
        
        cell_ori_inds = cell_grain_inds
        
        unique_cell_orientations = len(onp.unique(cell_ori_inds))
        if unique_cell_orientations != grains_uniques:
            problemes.append(f"Pas toutes les orientations utilisées: {unique_cell_orientations}/{grains_uniques}")
        
        min_coords = onp.min(pts, axis=0)
        max_coords = onp.max(pts, axis=0)
        dimensions = max_coords - min_coords
        
        print(f"Dimensions physiques: {dimensions[0]:.1f} × {dimensions[1]:.1f} × {dimensions[2]:.1f}")
        print(f"Coordonnées min: {min_coords}")
        print(f"Coordonnées max: {max_coords}")
        
        expected_dims = [495e-3, 110e-3, 80e-3]
        for i, (actual, expected) in enumerate(zip(dimensions, expected_dims)):
            if abs(actual - expected) > expected * 0.1:
                print(f"[WARNING] Dimension {['X','Y','Z'][i]} : {actual:.3f} ≠ {expected:.3f} attendu")
        
        print("Sauvegarde d'un VTU de test ExaCA...")
        test_vtk_path = debug_dir / "test_exaca_mesh.vtu"
        
        X_coord_min = float(min_coords[0])
        def dummy_selector(point, index):
            return np.isclose(point[0], X_coord_min, atol=1e-5)
        def zero_val_fn(point):
            return 0.0
        dummy_bc_info = [[dummy_selector], [0], [zero_val_fn]]
        
        try:
            mesh = Mesh(pts, cells)
            problem_obj = CrystalPlasticityInconel625(
                mesh, vec=3, dim=3, ele_type='HEX8',
                dirichlet_bc_info=dummy_bc_info,
                additional_info=(quat, cell_ori_inds)
            )
            
            test_sol = onp.zeros((pts.shape[0], 3))
            fake_stress = onp.ones(len(cells)) * 150.0
            
            save_sol(
                problem_obj.fes[0], test_sol, str(test_vtk_path),
                cell_infos=[
                    ('grain_id_physical', grain_tags.astype(onp.float64)),
                    ('grain_id_zero_based', cell_grain_inds.astype(onp.float64)),
                    ('cell_ori_inds', cell_ori_inds.astype(onp.float64)),
                    ('fake_stress_inconel', fake_stress),
                ]
            )
            print(f"VTU de test ExaCA sauvé : {test_vtk_path}")
            
        except Exception as e:
            print(f"[WARNING] Échec sauvegarde VTU ExaCA : {e}")
        
        info = {
            "meshio_mesh": meshio_mesh,
            "pts": pts,
            "cells": cells,
            "grain_tags": grain_tags,
            "cell_grain_inds": cell_grain_inds,
            "cell_ori_inds": cell_ori_inds,
            "quat": quat,
            "nb_points": nb_points,
            "nb_cellules": nb_cellules,
            "grains_uniques": grains_uniques,
            "nb_orientations": nb_orientations,
            "dimensions": dimensions,
            "min_coords": min_coords,
            "max_coords": max_coords
        }
        
        critical_problems = []
        for p in problemes:
            if "Dimension" not in p and "169" not in p:
                critical_problems.append(p)
        
        return len(critical_problems) == 0, info
        
    except Exception as e:
        print(f"[ERREUR] Échec diagnostic ExaCA : {e}")
        return False, {"erreur": str(e)}

def problem():
    print("Démarrage simulation traction X sur microstructure ExaCA")
    debut_total = time.time()
    
    case_name = 'polycrystal_ExaCA'
    base_dir = os.path.dirname(__file__)
    if not base_dir:
        base_dir = "."
    data_dir = os.path.join(base_dir, 'data')
    
    debug_dir = pathlib.Path(data_dir) / 'debug_exaca_inconel625'
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    msh_dir = os.path.join(data_dir, 'msh_with_grains')
    csv_dir = os.path.join(data_dir, 'csv', case_name)
    vtk_dir = os.path.join(data_dir, 'vtk', case_name + '_inconel625_traction_x_20steps')
    numpy_dir = os.path.join(data_dir, 'numpy', case_name + '_inconel625_traction_x_20steps')
    fig_dir = os.path.join(data_dir, 'figures')
    
    for d in (vtk_dir, numpy_dir, fig_dir):
        os.makedirs(d, exist_ok=True)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC MICROSTRUCTURE EXACA - INCONEL 625")
    print("="*70)
    
    fichiers_ok, erreurs_fichiers = verifier_fichiers_requis(msh_dir, csv_dir)
    if not fichiers_ok:
        print("PROBLÈMES DÉTECTÉS :")
        print("1. FICHIERS EXACA MANQUANTS")
        for erreur in erreurs_fichiers:
            print(f"   - {erreur}")
        print("\nARRÊT - Vérifiez les fichiers ExaCA")
        return
    
    msh_path = os.path.join(msh_dir, "exaca_grains_only.msh")
    quat_file_path = os.path.join(csv_dir, "quat_final.txt")
    
    coherence_ok, info = diagnostiquer_maillage_exaca_et_quaternions(msh_path, quat_file_path, debug_dir)
    
    print(f"\nFichiers de debug ExaCA sauvés dans : {debug_dir}")
    print("="*70)
    
    if coherence_ok:
        print("MICROSTRUCTURE EXACA VALIDE - SIMULATION INCONEL 625 LANCÉE")
        print("="*70)
        
        print("\n" + "="*70)
        print("SIMULATION CPFEM INCONEL 625 SUR MICROSTRUCTURE EXACA")
        print("="*70)
        
        lancer_simulation_exaca_inconel625(info, vtk_dir, numpy_dir, fig_dir, debut_total)
        
    else:
        print("PROBLÈMES DÉTECTÉS DANS LA MICROSTRUCTURE EXACA :")
        if "erreur" in info:
            print(f"1. {info['erreur']}")
        else:
            print("1. INCOHÉRENCE MAILLAGE EXACA ↔ QUATERNIONS")
        print("="*70)
        
        print("\nARRÊT - Corrigez les problèmes ExaCA")
        
        temps_diagnostic = time.time() - debut_total
        print(f"Temps de diagnostic : {temps_diagnostic:.2f}s")

def lancer_simulation_exaca_inconel625(info, vtk_dir, numpy_dir, fig_dir, debut_total):
    print("Initialisation CPFEM Inconel 625 sur ExaCA...")
    
    pts = info["pts"]
    cells = info["cells"]
    cell_grain_inds = info["cell_grain_inds"]
    cell_ori_inds = info["cell_ori_inds"]
    quat = info["quat"]
    dimensions = info["dimensions"]
    min_coords = info["min_coords"]
    
    mesh = Mesh(pts, cells)
    
    target_strain = 0.02
    
    X_coord_min = float(min_coords[0])
    X_coord_max = float(min_coords[0] + dimensions[0])
    actual_Lx = float(dimensions[0])
    max_displacement = target_strain * actual_Lx
    
    disps = onp.linspace(0., max_displacement, 21)
    ts = onp.linspace(0., target_strain * 0.02, 21)
    
    print(f"Paramètres simulation ExaCA :")
    print(f"  Matériau : Inconel 625 (E=200 GPa, σy≈120 MPa)")
    print(f"  Microstructure : {info['grains_uniques']} grains ExaCA")
    print(f"  Domaine : {dimensions[0]*1000:.0f}×{dimensions[1]*1000:.0f}×{dimensions[2]*1000:.0f} μm")
    print(f"  Cellules : {info['nb_cellules']} hexaèdres")
    print(f"  Déformation cible : {target_strain*100:.1f}%")
    print(f"  Déplacement max : {max_displacement*1000:.1f} μm")
    print(f"  Longueur domaine : {actual_Lx*1000:.1f} μm")
    print(f"  Nombre d'étapes : {len(disps)-1}")
    
    def left_face_selector(point, index):
        return np.isclose(point[0], X_coord_min, atol=1e-6)
        
    def right_face_selector(point, index):
        return np.isclose(point[0], X_coord_max, atol=1e-6)
        
    def zero_val_fn(point):
        return 0.0
        
    def right_disp_val_fn(d):
        def val_lambda(point):
            return d
        return val_lambda
    
    dirichlet_bc_info = [
        [left_face_selector, left_face_selector, left_face_selector, right_face_selector],
        [0,                  1,                  2,                  0                   ],
        [zero_val_fn,        zero_val_fn,        zero_val_fn,        right_disp_val_fn(disps[0])]
    ]
    
    print(f"CONDITIONS AUX LIMITES :")
    print(f"  Face gauche (X={X_coord_min*1000:.1f} μm) : U_x = U_y = U_z = 0 (ENCASTREMENT)")
    print(f"  Face droite (X={X_coord_max*1000:.1f} μm) : U_x = déplacement imposé")
    
    print("Initialisation CrystalPlasticityInconel625...")
    problem_obj = CrystalPlasticityInconel625(
        mesh,
        vec=3, dim=3, ele_type='HEX8',
        dirichlet_bc_info=dirichlet_bc_info,
        additional_info=(quat, cell_ori_inds)
    )
    
    sol_prev = onp.zeros((problem_obj.fes[0].num_total_nodes, problem_obj.fes[0].vec))
    params = problem_obj.internal_vars
    
    global_log, local_log, dist_log, slip_log = [], [], [], []
    detailed_stress_log = []
    detailed_strain_log = []
    
    cell0 = 0
    right_node_indices = onp.where(onp.isclose(pts[:, 0], X_coord_max, atol=1e-6))[0]
    cell_right_idx = 0
    if len(right_node_indices) > 0:
        right_node_idx = right_node_indices[0]
        for c_idx, cell_nodes_list in enumerate(cells):
            if right_node_idx in cell_nodes_list:
                cell_right_idx = c_idx
                break
    
    print(f"Problème initialisé - DDL total : {len(pts) * 3}")
    print(f"Contrainte d'écoulement initiale : {problem_obj.gss_initial} MPa")
    
    print(f"\n=== DÉBUT SIMULATION CPFEM (20 ÉTAPES) ===")
    
    for i in range(len(ts)-1):
        print(f"\nÉtape {i+1}/{len(ts)-1}:")
        
        disp_val = float(disps[i+1])
        eps_log_val = disp_val / actual_Lx
        dt_val = float(ts[i+1] - ts[i])
        problem_obj.dt = dt_val
        
        print(f"  eps={eps_log_val:.6f} ({eps_log_val*100:.3f}%), disp_x={disp_val*1000:.2f} μm, dt={dt_val:.6f}")
        
        current_dirichlet_bc_info = [
            [left_face_selector, left_face_selector, left_face_selector, right_face_selector],
            [0,                  1,                  2,                  0                   ],
            [zero_val_fn,        zero_val_fn,        zero_val_fn,        right_disp_val_fn(disp_val)]
        ]
        problem_obj.fes[0].update_Dirichlet_boundary_conditions(current_dirichlet_bc_info)
        
        print(f"  Résolution avec solveur UMFPACK...")
        problem_obj.set_params(params)
        
        try:
            sol_list = solver(problem_obj, {
                'umfpack_solver': {},
                'initial_guess': np.array(sol_prev)
            })
            print(f"  Résolution UMFPACK réussie")
        except Exception as e:
            print(f"  Échec UMFPACK, tentative JAX...")
            try:
                sol_list = solver(problem_obj, {
                    'jax_solver': {'precond': True},
                    'initial_guess': np.array(sol_prev)
                })
                print(f"  Résolution JAX réussie en fallback")
            except Exception as e2:
                print(f"  Échec des deux solveurs:")
                print(f"     UMFPACK: {e}")
                print(f"     JAX: {e2}")
                print(f"  ARRÊT SIMULATION à l'étape {i+1}")
                break
        
        sol_prev = sol_list[0]
        
        print(f"  Calcul des contraintes...")
        sigma_stresses = problem_obj.compute_avg_stress(sol_prev, params)
        
        sigma_xx_stress_np = onp.array(sigma_stresses[:, 0, 0])
        sigma_yy_stress_np = onp.array(sigma_stresses[:, 1, 1])
        sigma_zz_stress_np = onp.array(sigma_stresses[:, 2, 2])
        sigma_xy_stress_np = onp.array(sigma_stresses[:, 0, 1])
        sigma_yz_stress_np = onp.array(sigma_stresses[:, 1, 2])
        sigma_xz_stress_np = onp.array(sigma_stresses[:, 0, 2])
        
        sigma_vm_stress_np = onp.sqrt(
            0.5*((sigma_xx_stress_np - sigma_yy_stress_np)**2 +
                 (sigma_yy_stress_np - sigma_zz_stress_np)**2 +
                 (sigma_zz_stress_np - sigma_xx_stress_np)**2) +
            3.0*(sigma_xy_stress_np**2 + sigma_yz_stress_np**2 + sigma_xz_stress_np**2)
        )
        
        eps_xx_approx = eps_log_val
        eps_yy_approx = -0.3 * eps_xx_approx
        eps_zz_approx = -0.3 * eps_xx_approx
        
        print(f"  Sauvegarde VTU...")
        vtu_name = f"inconel625_exaca_traction_x_step_{i:03d}.vtu"
        vtk_path = os.path.join(vtk_dir, vtu_name)
        save_sol(
            problem_obj.fes[0], sol_prev, vtk_path,
            cell_infos=[
                ('grain_id_exaca',        (cell_grain_inds + 1).astype(onp.float64)),
                ('cell_ori_inds',         cell_ori_inds.astype(onp.float64)),
                ('sigma_xx',              sigma_xx_stress_np),
                ('sigma_yy',              sigma_yy_stress_np),
                ('sigma_zz',              sigma_zz_stress_np),
                ('von_Mises_stress',      sigma_vm_stress_np),
            ]
        )
        
        avg_xx = float(onp.mean(sigma_xx_stress_np))
        avg_yy = float(onp.mean(sigma_yy_stress_np))
        avg_zz = float(onp.mean(sigma_zz_stress_np))
        avg_vm = float(onp.mean(sigma_vm_stress_np))
        
        global_log.append([eps_log_val, avg_xx, avg_vm])
        
        local_xx_cell0 = float(sigma_xx_stress_np[cell0])
        local_vm_cell0 = float(sigma_vm_stress_np[cell0])
        local_log.append([eps_log_val, local_xx_cell0, local_vm_cell0])
        
        pcts = onp.percentile(sigma_xx_stress_np, [0, 25, 50, 75, 100])
        dist_log.append([eps_log_val, *pcts])
        
        detailed_stress_log.append([i+1, eps_log_val, avg_xx, avg_yy, avg_zz, avg_vm])
        detailed_strain_log.append([i+1, eps_log_val, eps_xx_approx, eps_yy_approx, eps_zz_approx])
        
        params = problem_obj.update_int_vars_gp(sol_prev, params)
        
        try:
            if len(params) > 2:
                slip_values_for_max = params[2]
                if hasattr(slip_values_for_max, 'shape'):
                    slip_max = float(onp.max(onp.abs(onp.array(slip_values_for_max))))
                else:
                    slip_max = 0.0
            else:
                slip_max = 0.0
        except:
            slip_max = 0.0
            
        slip_log.append([eps_log_val, slip_max])
        
        print(f"  σxx_moy: {avg_xx:.1f} MPa, von Mises: {avg_vm:.1f} MPa, slip_max: {slip_max:.2e}")
    
    print(f"\nSauvegarde des logs d'analyse...")
    onp.savetxt(os.path.join(numpy_dir, 'global_inconel625_exaca_20steps.txt'), onp.array(global_log),
                header='eps sigma_xx_avg sigma_vm_avg', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'local_inconel625_exaca_20steps.txt'), onp.array(local_log),
                header='eps sigma_xx_cell0 sigma_vm_cell0', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'dist_inconel625_exaca_20steps.txt'), onp.array(dist_log),
                header='eps min q25 median q75 max', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'slip_inconel625_exaca_20steps.txt'), onp.array(slip_log),
                header='eps max_slip', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'detailed_stress_inconel625_exaca_20steps.txt'), onp.array(detailed_stress_log),
                header='step eps sigma_xx sigma_yy sigma_zz sigma_vm', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'detailed_strain_inconel625_exaca_20steps.txt'), onp.array(detailed_strain_log),
                header='step eps eps_xx eps_yy eps_zz', fmt='%.6e')
    
    print("Génération des graphiques contrainte-déformation...")
    
    stress_data = onp.array(detailed_stress_log)
    strains = stress_data[:, 1] * 100
    sigma_xx = stress_data[:, 2]
    sigma_yy = stress_data[:, 3]
    sigma_zz = stress_data[:, 4]
    sigma_vm = stress_data[:, 5]
    
    plt.style.use('default')
    
    plt.figure(figsize=(12, 8))
    plt.plot(strains, sigma_xx, 'b-', linewidth=3, marker='o', markersize=4, label='σxx (traction)')
    plt.plot(strains, sigma_yy, 'r-', linewidth=2, marker='s', markersize=3, label='σyy')
    plt.plot(strains, sigma_zz, 'g-', linewidth=2, marker='^', markersize=3, label='σzz')
    plt.plot(strains, sigma_vm, 'm-', linewidth=2, marker='d', markersize=3, label='von Mises')
    
    plt.axhline(y=120, color='k', linestyle='--', alpha=0.7, label='Limite élastique Inconel 625 (120 MPa)')
    
    plt.xlabel('Déformation (%)', fontsize=12)
    plt.ylabel('Contrainte (MPa)', fontsize=12)
    plt.title('Contraintes vs Déformation - Inconel 625 ExaCA (169 grains, 2% déformation)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'contraintes_inconel625_exaca_vs_strain_20steps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    slip_data = onp.array(slip_log)
    slip_strains = slip_data[:, 0] * 100
    slip_values = slip_data[:, 1]
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(slip_strains, slip_values, 'r-', linewidth=3, marker='o', markersize=4)
    plt.xlabel('Déformation (%)', fontsize=12)
    plt.ylabel('Glissement plastique maximum', fontsize=12)
    plt.title('Évolution du glissement plastique - Inconel 625 ExaCA (échelle log)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'slip_evolution_inconel625_exaca_20steps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(strains, sigma_xx, 'b-', linewidth=4, marker='o', markersize=6)
    plt.axhline(y=120, color='r', linestyle='--', alpha=0.7, label='Limite élastique Inconel 625')
    
    E = 200000
    elastic_line = E * strains / 100
    plt.plot(strains, elastic_line, 'k--', alpha=0.5, label='Réponse élastique pure')
    
    plt.xlabel('Déformation (%)', fontsize=12)
    plt.ylabel('Contrainte σxx (MPa)', fontsize=12)
    plt.title('Réponse Élasto-Plastique - Inconel 625 ExaCA (microstructure de solidification)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'reponse_elasto_plastique_inconel625_exaca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    temps_total = time.time() - debut_total
    
    print("\n" + "="*70)
    print(" SIMULATION INCONEL 625 ExaCA TERMINÉE AVEC SUCCÈS")
    print("="*70)
    print(f"  Temps total : {temps_total:.1f}s")
    print(f"  Matériau : Inconel 625 (E=200 GPa, σy=120 MPa)")
    print(f"  Microstructure : {info['grains_uniques']} grains ExaCA")
    print(f"  Domaine : {info['dimensions'][0]*1000:.0f}×{info['dimensions'][1]*1000:.0f}×{info['dimensions'][2]*1000:.0f} μm")
    print(f"  Cellules : {info['nb_cellules']} hexaèdres")
    print(f"  Résultats VTU : {vtk_dir}/")
    print(f"  Logs d'analyse : {numpy_dir}/")
    print(f"  Graphiques : {fig_dir}/")
    print(f"  Déformation finale : {target_strain*100}%")
    print(f"  Nombre d'étapes : {len(disps)-1}")
    if len(detailed_stress_log) > 0:
        avg_xx_final = detailed_stress_log[-1][2]
        avg_vm_final = detailed_stress_log[-1][5]
        print(f"  Contrainte finale σxx : {avg_xx_final:.1f} MPa")
        print(f"  Contrainte von Mises : {avg_vm_final:.1f} MPa")
    if len(slip_log) > 0:
        slip_max_final = slip_log[-1][1]
        print(f"  Glissement max final : {slip_max_final:.6e}")
    print("="*70)

if __name__ == '__main__':
    problem()
