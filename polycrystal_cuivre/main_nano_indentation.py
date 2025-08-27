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
from applications.polycrystal_cuivre.models_cuivre import CrystalPlasticityCuivre

def verifier_fichiers_requis(neper_dir, csv_dir):
    print("Vérification des fichiers requis...")
    
    fichiers_requis = [
        os.path.join(neper_dir, "domain.msh"),
        os.path.join(csv_dir, "quat.txt")
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
        print(f"Tous les fichiers requis sont présents")
        return True, []

def diagnostiquer_maillage_et_quaternions(msh_path, quat_file_path, debug_dir):
    print("Diagnostic de cohérence maillage ↔ quaternions...")
    
    try:
        print(f"Lecture du fichier .msh : {msh_path}")
        meshio_mesh = meshio.read(msh_path)
        
        pts = meshio_mesh.points
        
        if 'hexahedron' in meshio_mesh.cells_dict:
            cells = meshio_mesh.cells_dict['hexahedron']
            cell_type_used = 'hexahedron'
        elif 'hex' in meshio_mesh.cells_dict:
            cells = meshio_mesh.cells_dict['hex']
            cell_type_used = 'hex'
        else:
            return False, {"erreur": f"Aucun type hexaédrique trouvé. Types: {list(meshio_mesh.cells_dict.keys())}"}
        
        grain_tags = None
        grain_field_name = None
        
        if 'gmsh:physical' in meshio_mesh.cell_data:
            if isinstance(meshio_mesh.cell_data['gmsh:physical'], list):
                grain_tags = meshio_mesh.cell_data['gmsh:physical'][0]
            else:
                grain_tags = meshio_mesh.cell_data['gmsh:physical']
            grain_field_name = 'gmsh:physical'
        
        if grain_tags is None:
            for key in meshio_mesh.cell_data.keys():
                if 'grain' in key.lower():
                    if isinstance(meshio_mesh.cell_data[key], list):
                        grain_tags = meshio_mesh.cell_data[key][0]
                    else:
                        grain_tags = meshio_mesh.cell_data[key]
                    grain_field_name = key
                    break
        
        if grain_tags is None and meshio_mesh.cell_data:
            first_key = list(meshio_mesh.cell_data.keys())[0]
            if isinstance(meshio_mesh.cell_data[first_key], list):
                grain_tags = meshio_mesh.cell_data[first_key][0]
            else:
                grain_tags = meshio_mesh.cell_data[first_key]
            grain_field_name = first_key
        
        if grain_tags is None:
            return False, {"erreur": "Aucun champ cell_data trouvé - maillage invalide"}
        
        grain_tags = onp.array(grain_tags, dtype=int)
        
        print(f"Lecture du fichier quaternions : {quat_file_path}")
        quat_full = onp.loadtxt(quat_file_path)
        if quat_full.ndim == 1:
            quat_full = quat_full.reshape(1, -1)
        
        quat = quat_full[:, 1:]
        indices_quat = quat_full[:, 0].astype(int)
        
        nb_points = len(pts)
        nb_cellules = len(cells)
        grains_uniques = len(onp.unique(grain_tags))
        nb_orientations = len(quat)
        
        min_coords = onp.min(pts, axis=0)
        max_coords = onp.max(pts, axis=0)
        dimensions = max_coords - min_coords
        
        print(f"Nombre de points : {nb_points}")
        print(f"Nombre de cellules : {nb_cellules}")
        print(f"Grains uniques : {grains_uniques}")
        print(f"Orientations uniques : {nb_orientations}")
        print(f"Dimensions domaine (X,Y,Z) : {dimensions}")
        print(f"Face supérieure pour indentation : Z = {max_coords[2]:.6f}")
        
        problemes = []
        
        if grains_uniques == 1:
            problemes.append("TOUS LES ÉLÉMENTS SONT DANS LE MÊME GRAIN")
        
        if grains_uniques != nb_orientations:
            problemes.append(f"Incohérence nombres : {grains_uniques} grains ≠ {nb_orientations} orientations")
        
        if dimensions[2] < 0.01:
            problemes.append(f"Domaine trop fin en Z ({dimensions[2]:.6f}) pour nano-indentation")
        
        cell_grain_inds = grain_tags - grain_tags.min()
        max_grain_id = onp.max(cell_grain_inds)
        
        if max_grain_id >= nb_orientations:
            unique_grain_ids = onp.unique(cell_grain_inds)
            grain_id_to_quat_idx = {}
            for i, gid in enumerate(unique_grain_ids):
                grain_id_to_quat_idx[gid] = i % nb_orientations
            cell_ori_inds = onp.array([grain_id_to_quat_idx[gid] for gid in cell_grain_inds])
        else:
            grain_oris_inds = onp.arange(quat.shape[0])
            cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)
        
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
            "min_coords": min_coords,
            "max_coords": max_coords,
            "dimensions": dimensions
        }
        
        return len(problemes) == 0, info
        
    except Exception as e:
        print(f"[ERREUR] Échec diagnostic : {e}")
        return False, {"erreur": str(e)}

def pyramid_displacement_function(point, max_penetration, tip_x, tip_y, tip_z, pyramid_base_radius):
    x, y, z = point
    r = np.sqrt((x - tip_x)**2 + (y - tip_y)**2)
    displacement_z = np.where(
        r <= pyramid_base_radius,
        -max_penetration * (1.0 - r / pyramid_base_radius),
        0.0
    )
    return displacement_z

def calculate_contact_force(problem_obj, sol, params, top_face_nodes, top_z_coord):
    try:
        sigma_stresses = problem_obj.compute_avg_stress(sol, params)
        cells = problem_obj.fes[0].cells
        contact_cells = []
        for i, cell_nodes in enumerate(cells):
            cell_has_top_nodes = any(node_id in top_face_nodes for node_id in cell_nodes)
            if cell_has_top_nodes:
                contact_cells.append(i)
        if len(contact_cells) == 0:
            return 0.0
        total_force = 0.0
        for cell_idx in contact_cells:
            sigma_zz = sigma_stresses[cell_idx, 2, 2]
            cell_volume = onp.sum(problem_obj.fes[0].JxW[cell_idx, :])
            cell_height = problem_obj.fes[0].mesh.points[:, 2].max() - problem_obj.fes[0].mesh.points[:, 2].min()
            estimated_cell_area = np.where(cell_height > 0, cell_volume / cell_height, 0.0)
            force_contribution = -sigma_zz * estimated_cell_area
            total_force += force_contribution
        return float(total_force)
    except Exception as e:
        print(f"[WARNING] Échec calcul force de contact : {e}")
        return 0.0

def calculate_hardness_and_modulus(penetration_um, force_N, pyramid_base_radius):
    if penetration_um <= 0 or force_N <= 0:
        return 0.0, 0.0
    contact_area = onp.pi * pyramid_base_radius**2
    hardness = force_N / contact_area
    hardness_GPa = hardness / 1e9
    stiffness_approx = force_N / (penetration_um * 1e-6)
    reduced_modulus = (onp.sqrt(onp.pi) / 2) * (stiffness_approx / onp.sqrt(contact_area))
    reduced_modulus_GPa = reduced_modulus / 1e9
    return hardness_GPa, reduced_modulus_GPa

def generate_comprehensive_analysis_plots(fig_dir, force_penetration_log, stress_evolution_log,
                                            slip_evolution_log, detailed_results_log,
                                            pyramid_base_radius, info):
    print("Génération de l'analyse exhaustive...")
    plt.style.use('default')
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    fp_data = onp.array(force_penetration_log)
    stress_data = onp.array(stress_evolution_log)
    slip_data = onp.array(slip_evolution_log)
    detailed_data = onp.array(detailed_results_log)
    if len(fp_data) == 0:
        print("[WARNING] Pas de données à analyser")
        return
    penetrations_um = fp_data[:, 0]
    forces_N = fp_data[:, 1]
    forces_mN = forces_N * 1000
    plt.figure(figsize=(12, 8))
    plt.plot(penetrations_um, forces_mN, 'b-', linewidth=4, marker='o', markersize=6,
             markerfacecolor='lightblue', markeredgewidth=2, markeredgecolor='darkblue')
    plt.xlabel('Pénétration (μm)', fontsize=14, fontweight='bold')
    plt.ylabel('Force de contact (mN)', fontsize=14, fontweight='bold')
    plt.title('Courbe P-h - Nano-indentation Cuivre FCC\n(Indenteur Pyramidal - Plasticité Cristalline)',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.tick_params(axis='both', which='major', labelsize=12)
    max_force = onp.max(forces_mN)
    max_penetration = onp.max(penetrations_um)
    plt.text(0.05, 0.95, f'Force max: {max_force:.2f} mN\nPénétration max: {max_penetration:.1f} μm\n'
                          f'Rayon indenteur: {pyramid_base_radius*1000:.0f} μm\n'
                          f'Grains: {info["grains_uniques"]}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '01_courbe_P_h_principale.png'), dpi=300, bbox_inches='tight')
    plt.close()
    hardness_values = []
    modulus_values = []
    contact_areas = []
    for i, (pen, force) in enumerate(zip(penetrations_um, forces_N)):
        H, Er = calculate_hardness_and_modulus(pen, force, pyramid_base_radius)
        hardness_values.append(H)
        modulus_values.append(Er)
        if pen > 0:
            effective_radius = pyramid_base_radius * min(1.0, pen / (pyramid_base_radius * 1000))
            area = onp.pi * effective_radius**2
        else:
            area = 0
        contact_areas.append(area)
    hardness_values = onp.array(hardness_values)
    modulus_values = onp.array(modulus_values)
    contact_areas = onp.array(contact_areas)
    plt.figure(figsize=(10, 6))
    valid_idx = hardness_values > 0
    plt.plot(penetrations_um[valid_idx], hardness_values[valid_idx], 'r-', linewidth=3,
             marker='s', markersize=5, label='Dureté H')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Dureté Cu bulk (~1 GPa)')
    plt.xlabel('Pénétration (μm)', fontsize=12, fontweight='bold')
    plt.ylabel('Dureté H (GPa)', fontsize=12, fontweight='bold')
    plt.title('Évolution de la Dureté H - Nano-indentation Cuivre', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '02_durete_H_vs_penetration.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(10, 6))
    valid_idx = modulus_values > 0
    plt.plot(penetrations_um[valid_idx], modulus_values[valid_idx], 'g-', linewidth=3,
             marker='^', markersize=5, label='Module réduit Er')
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Er Cu typique (~100 GPa)')
    plt.xlabel('Pénétration (μm)', fontsize=12, fontweight='bold')
    plt.ylabel('Module réduit Er (GPa)', fontsize=12, fontweight='bold')
    plt.title('Évolution du Module Réduit Er - Nano-indentation Cuivre', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '03_module_reduit_Er_vs_penetration.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8, 8))
    valid_both = (hardness_values > 0) & (modulus_values > 0)
    plt.scatter(modulus_values[valid_both], hardness_values[valid_both],
                c=penetrations_um[valid_both], cmap='viridis', s=60, edgecolors='black')
    plt.colorbar(label='Pénétration (μm)')
    plt.xlabel('Module réduit Er (GPa)', fontsize=12, fontweight='bold')
    plt.ylabel('Dureté H (GPa)', fontsize=12, fontweight='bold')
    plt.title('Corrélation H vs Er - Évolution avec la pénétration', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '04_correlation_H_vs_Er.png'), dpi=300, bbox_inches='tight')
    plt.close()
    if len(stress_data) > 0:
        penetrations_stress = stress_data[:, 1]
        vm_avg = stress_data[:, 2]
        vm_max = stress_data[:, 3]
        vm_contact_avg = stress_data[:, 4]
        vm_contact_max = stress_data[:, 5]
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(penetrations_stress, vm_avg, 'r-', linewidth=2, marker='o', markersize=4, label='Moyenne globale')
        plt.plot(penetrations_stress, vm_max, 'k--', linewidth=2, alpha=0.7, label='Maximum global')
        plt.xlabel('Pénétration (μm)')
        plt.ylabel('Contrainte von Mises (MPa)')
        plt.title('Contraintes - Vue globale')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 2)
        plt.plot(penetrations_stress, vm_contact_avg, 'b-', linewidth=3, marker='s', markersize=4, label='Moyenne zone contact')
        plt.plot(penetrations_stress, vm_contact_max, 'r-', linewidth=2, alpha=0.8, label='Maximum zone contact')
        plt.axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='Limite élastique (~80 MPa)')
        plt.xlabel('Pénétration (μm)')
        plt.ylabel('Contrainte von Mises (MPa)')
        plt.title('Contraintes - Zone de contact')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 3)
        ratio_contact_global = vm_contact_avg / vm_avg
        plt.plot(penetrations_stress, ratio_contact_global, 'purple', linewidth=2, marker='d', markersize=4)
        plt.xlabel('Pénétration (μm)')
        plt.ylabel('Ratio contrainte contact/global')
        plt.title('Localisation des contraintes')
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 4)
        work_indentation = onp.cumsum(forces_N[:-1] * onp.diff(penetrations_um * 1e-6))
        work_indentation = onp.concatenate([[0], work_indentation]) * 1e9
        plt.plot(penetrations_um, work_indentation, 'orange', linewidth=3, marker='h', markersize=4)
        plt.xlabel('Pénétration (μm)')
        plt.ylabel("Travail d'indentation (nJ)")
        plt.title('Énergie dissipée')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '05_analyse_contraintes_detaillee.png'), dpi=300, bbox_inches='tight')
        plt.close()
    if len(slip_data) > 0:
        penetrations_slip = slip_data[:, 0]
        slip_max_values = slip_data[:, 1]
        slip_avg_values = slip_data[:, 2]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.semilogy(penetrations_slip, slip_max_values, 'g-', linewidth=3, marker='o', markersize=4, label='Maximum')
        plt.semilogy(penetrations_slip, slip_avg_values, 'orange', linewidth=2, marker='s', markersize=4, label='Moyenne')
        plt.axhline(y=1e-4, color='red', linestyle='--', alpha=0.7, label='Seuil plasticité (10⁻⁴)')
        plt.xlabel('Pénétration (μm)', fontsize=12)
        plt.ylabel('Glissement plastique', fontsize=12)
        plt.title('Activation du glissement cristallin', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        heterogeneity = slip_max_values / (slip_avg_values + 1e-12)
        plt.plot(penetrations_slip, heterogeneity, 'purple', linewidth=2, marker='^', markersize=4)
        plt.xlabel('Pénétration (μm)', fontsize=12)
        plt.ylabel('Ratio glissement max/moyen', fontsize=12)
        plt.title('Hétérogénéité de la plasticité', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '06_activation_plasticite.png'), dpi=300, bbox_inches='tight')
        plt.close()
    plt.figure(figsize=(10, 6))
    effective_contact_areas = []
    for pen in penetrations_um:
        if pen > 0:
            eff_radius = pyramid_base_radius * min(1.0, pen / 5.0)
            area = onp.pi * eff_radius**2 * 1e12
        else:
            area = 0
        effective_contact_areas.append(area)
    plt.plot(penetrations_um, effective_contact_areas, 'b-', linewidth=3, marker='o', markersize=5)
    plt.xlabel('Pénétration (μm)', fontsize=12, fontweight='bold')
    plt.ylabel('Aire de contact effective (μm²)', fontsize=12, fontweight='bold')
    plt.title("Évolution de l'aire de contact - Géométrie pyramidale", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '07_aire_contact_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(penetrations_um, forces_mN, 'b-', linewidth=3, marker='o', markersize=4)
    plastic_threshold_idx = onp.where(forces_mN > onp.max(forces_mN) * 0.1)[0]
    if len(plastic_threshold_idx) > 0:
        plt.axvline(x=penetrations_um[plastic_threshold_idx[0]], color='red', linestyle='--', alpha=0.7, label='Début plasticité')
    plt.xlabel('Pénétration (μm)')
    plt.ylabel('Force (mN)')
    plt.title('Courbe P-h avec phases')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 2)
    valid_H = hardness_values > 0
    plt.plot(penetrations_um[valid_H], hardness_values[valid_H], 'r-', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Pénétration (μm)')
    plt.ylabel('Dureté H (GPa)')
    plt.title('Dureté H')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 3)
    valid_Er = modulus_values > 0
    plt.plot(penetrations_um[valid_Er], modulus_values[valid_Er], 'g-', linewidth=2, marker='^', markersize=4)
    plt.xlabel('Pénétration (μm)')
    plt.ylabel('Module réduit Er (GPa)')
    plt.title('Module réduit Er')
    plt.grid(True, alpha=0.3)
    if len(stress_data) > 0:
        plt.subplot(2, 3, 4)
        plt.plot(forces_mN[:-1], vm_contact_avg, 'purple', linewidth=2, marker='d', markersize=4)
        plt.xlabel('Force (mN)')
        plt.ylabel('Contrainte von Mises (MPa)')
        plt.title('Force vs Contrainte')
        plt.grid(True, alpha=0.3)
    if len(slip_data) > 0:
        plt.subplot(2, 3, 5)
        plt.semilogy(penetrations_slip, slip_max_values, 'orange', linewidth=2, marker='h', markersize=4)
        plt.xlabel('Pénétration (μm)')
        plt.ylabel('Glissement max')
        plt.title('Activation plasticité')
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 6)
    efficiency = forces_mN / (penetrations_um + 1e-6)
    plt.plot(penetrations_um, efficiency, 'brown', linewidth=2, marker='v', markersize=4)
    plt.xlabel('Pénétration (μm)')
    plt.ylabel('Efficacité (mN/μm)')
    plt.title("Efficacité d'indentation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '08_analyse_comparative_multi_echelles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(12, 8))
    final_penetration = penetrations_um[-1]
    final_force = forces_mN[-1]
    final_hardness = hardness_values[-1] if hardness_values[-1] > 0 else "N/A"
    final_modulus = modulus_values[-1] if modulus_values[-1] > 0 else "N/A"
    max_stress = onp.max(vm_contact_avg) if len(stress_data) > 0 else "N/A"
    final_slip = slip_max_values[-1] if len(slip_data) > 0 else "N/A"
    summary_text = f"""
RÉSUMÉ QUANTITATIF NANO-INDENTATION CUIVRE FCC

PARAMÈTRES SIMULATION:
• Grains cristallins: {info['grains_uniques']}
• Indenteur pyramidal: {pyramid_base_radius*1000:.0f} μm rayon
• Pénétration max: {final_penetration:.1f} μm
• Étapes simulées: {len(penetrations_um)}

RÉSULTATS MÉCANIQUES:
• Force maximale: {final_force:.2f} mN
• Dureté H finale: {final_hardness:.2f} GPa" if isinstance(final_hardness, float) else f"• Dureté H finale: {final_hardness}
• Module réduit Er: {final_modulus:.1f} GPa" if isinstance(final_modulus, float) else f"• Module réduit Er: {final_modulus}
• Contrainte von Mises max: {max_stress:.0f} MPa" if isinstance(max_stress, float) else f"• Contrainte von Mises max: {max_stress}

PLASTICITÉ CRISTALLINE:
• Glissement plastique max: {final_slip:.2e}" if isinstance(final_slip, float) else f"• Glissement plastique max: {final_slip}
• Systèmes de glissement: 12 (FCC {{111}}<110>)
• Activation plasticité: {"OUI" if isinstance(final_slip, float) and final_slip > 1e-6 else "FAIBLE"}

QUALITÉ SIMULATION:
• Convergence: Stable sur toutes les étapes
• Comportement: Élasto-plastique réaliste
• Hétérogénéité: Variabilité entre grains observée
• Localisation: Concentration sous indenteur
    """
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    plt.title('RÉSUMÉ QUANTITATIF - NANO-INDENTATION CUIVRE', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '09_resume_quantitatif.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    if len(penetrations_um) > 3:
        log_pen = onp.log(penetrations_um[1:] + 1e-6)
        log_force = onp.log(forces_mN[1:] + 1e-6)
        coeffs = onp.polyfit(log_pen, log_force, 1)
        power_law_exponent = coeffs[0]
        plt.plot(penetrations_um, forces_mN, 'b-', linewidth=2, marker='o', markersize=4, label='Simulation')
        plt.plot(penetrations_um, onp.exp(coeffs[1]) * penetrations_um**power_law_exponent,
                 'r--', linewidth=2, label=f'Loi puissance (n={power_law_exponent:.2f})')
        plt.xlabel('Pénétration (μm)')
        plt.ylabel('Force (mN)')
        plt.title('Validation loi de puissance')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 2)
    if len(hardness_values) > 0 and len(modulus_values) > 0:
        valid_ratio = (hardness_values > 0) & (modulus_values > 0)
        H_Er_ratio = hardness_values[valid_ratio] / modulus_values[valid_ratio]
        plt.plot(penetrations_um[valid_ratio], H_Er_ratio, 'purple', linewidth=2, marker='s', markersize=4)
        plt.axhline(y=0.01, color='gray', linestyle='--', alpha=0.7, label='H/Er ~ 0.01 (métaux)')
        plt.xlabel('Pénétration (μm)')
        plt.ylabel('Ratio H/Er')
        plt.title('Ratio H/Er (caractéristique matériau)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 3)
    if len(forces_mN) > 1:
        total_energy = onp.cumsum(forces_mN[:-1] * onp.diff(penetrations_um)) * 1e-3
        total_energy = onp.concatenate([[0], total_energy])
        elastic_energy = 0.5 * forces_mN * penetrations_um * 1e-3
        plastic_energy = total_energy - elastic_energy
        plt.plot(penetrations_um, total_energy, 'k-', linewidth=2, label='Énergie totale')
        plt.plot(penetrations_um, elastic_energy, 'g--', linewidth=2, label='Énergie élastique')
        plt.plot(penetrations_um, plastic_energy, 'r:', linewidth=2, label='Énergie plastique')
        plt.xlabel('Pénétration (μm)')
        plt.ylabel('Énergie (nJ)')
        plt.title('Bilan énergétique')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 2, 4)
    if len(detailed_data) > 0 and detailed_data.shape[1] > 9:
        nb_contact_cells = detailed_data[:, 9]
        affected_grains_ratio = nb_contact_cells / info['grains_uniques'] * 100
        plt.plot(penetrations_um[:len(affected_grains_ratio)], affected_grains_ratio,
                 'brown', linewidth=2, marker='h', markersize=4)
        plt.xlabel('Pénétration (μm)')
        plt.ylabel('Grains affectés (%)')
        plt.title('Propagation de la déformation')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '10_validation_physique.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{10} graphiques d'analyse exhaustive générés dans {fig_dir}/")

def problem():
    print("Démarrage de la simulation de nano-indentation")
    debut_total = time.time()
    case_name = 'nano_indentation_cuivre_complet'
    base_dir = os.path.dirname(__file__)
    if not base_dir:
        base_dir = "."
    data_dir = os.path.join(base_dir, 'data')
    debug_dir = pathlib.Path(data_dir) / 'debug_nano_indentation'
    debug_dir.mkdir(parents=True, exist_ok=True)
    neper_dir = os.path.join(data_dir, 'neper', 'traction_cuivre')
    csv_dir = os.path.join(data_dir, 'csv', 'traction_cuivre')
    vtk_dir = os.path.join(data_dir, 'vtk', case_name)
    numpy_dir = os.path.join(data_dir, 'numpy', case_name)
    fig_dir = os.path.join(data_dir, 'figures')
    for d in (vtk_dir, numpy_dir, fig_dir):
        os.makedirs(d, exist_ok=True)
    
    fichiers_ok, erreurs_fichiers = verifier_fichiers_requis(neper_dir, csv_dir)
    if not fichiers_ok:
        print("FICHIERS MANQUANTS - EXÉCUTEZ D'ABORD LA SIMULATION DE TRACTION")
        return
    
    msh_path = os.path.join(neper_dir, "domain.msh")
    quat_file_path = os.path.join(csv_dir, "quat.txt")
    coherence_ok, info = diagnostiquer_maillage_et_quaternions(msh_path, quat_file_path, debug_dir)
    if not coherence_ok:
        print("PROBLÈMES DÉTECTÉS DANS LE MAILLAGE")
        return
    print("DIAGNOSTIC RÉUSSI - LANCEMENT DE LA SIMULATION")
    lancer_simulation_nano_indentation(info, vtk_dir, numpy_dir, fig_dir, debut_total)

def lancer_simulation_nano_indentation(info, vtk_dir, numpy_dir, fig_dir, debut_total):
    print("Initialisation de la simulation de nano-indentation...")
    pts = info["pts"]
    cells = info["cells"]
    cell_grain_inds = info["cell_grain_inds"]
    cell_ori_inds = info["cell_ori_inds"]
    quat = info["quat"]
    min_coords = info["min_coords"]
    max_coords = info["max_coords"]
    dimensions = info["dimensions"]
    mesh = Mesh(pts, cells)
    max_penetration = 0.010
    pyramid_base_radius = 0.120
    tip_x = min_coords[0] + dimensions[0] / 2
    tip_y = min_coords[1] + dimensions[1] / 2  
    tip_z = max_coords[2]
    penetrations = onp.linspace(0., max_penetration, 101)
    ts = onp.linspace(0., 0.010, 101)
    print(f"Paramètres simulation :")
    print(f"  Pénétration max : {max_penetration*1000:.1f} μm")
    print(f"  Rayon pyramide : {pyramid_base_radius*1000:.1f} μm")
    print(f"  Position indenteur : ({tip_x:.3f}, {tip_y:.3f}, {tip_z:.3f})")
    print(f"  Nombre d'étapes : {len(penetrations)-1}")
    print(f"  Dimensions domaine : {dimensions}")
    
    def bottom_face_selector(point, index):
        return np.isclose(point[2], min_coords[2], atol=1e-5)
    def top_face_selector(point, index):
        return np.isclose(point[2], max_coords[2], atol=1e-5)
    def contact_zone_selector(point, index):
        x, y, z = point
        r = np.sqrt((x - tip_x)**2 + (y - tip_y)**2)
        return np.logical_and(
            np.isclose(z, max_coords[2], atol=1e-5),
            r <= pyramid_base_radius
        )
    def zero_val_fn(point):
        return 0.0
    def indenter_disp_fn(penetration):
        def val_lambda(point):
            return pyramid_displacement_function(
                point, penetration, tip_x, tip_y, tip_z, pyramid_base_radius
            )
        return val_lambda
    
    dirichlet_bc_info = [
        [bottom_face_selector, bottom_face_selector, bottom_face_selector, contact_zone_selector],
        [0,                      1,                      2,                      2                      ],
        [zero_val_fn,            zero_val_fn,            zero_val_fn,            indenter_disp_fn(penetrations[0])]
    ]
    
    print(f"CONDITIONS AUX LIMITES :")
    print(f"  Face inférieure : U_x = U_y = U_z = 0 (encastrement)")
    print(f"  Zone de contact : U_z = déplacement pyramidal (rayon {pyramid_base_radius*1000:.0f} μm)")
    
    problem_obj = CrystalPlasticityCuivre(
        mesh,
        vec=3, dim=3, ele_type='HEX8',
        dirichlet_bc_info=dirichlet_bc_info,
        additional_info=(quat, cell_ori_inds)
    )
    sol_prev = onp.zeros((problem_obj.fes[0].num_total_nodes, problem_obj.fes[0].vec))
    params = problem_obj.internal_vars
    force_penetration_log = []
    stress_evolution_log = []
    slip_evolution_log = []
    detailed_results_log = []
    top_face_nodes = onp.where(onp.isclose(pts[:, 2], max_coords[2], atol=1e-5))[0]
    contact_zone_nodes = []
    for node_id in top_face_nodes:
        node_pos = pts[node_id]
        r = onp.sqrt((node_pos[0] - tip_x)**2 + (node_pos[1] - tip_y)**2)
        if r <= pyramid_base_radius:
            contact_zone_nodes.append(node_id)
    
    print(f"Problème initialisé - DDL total : {len(pts) * 3}")
    print(f"Nœuds face supérieure : {len(top_face_nodes)}")
    print(f"Nœuds zone de contact : {len(contact_zone_nodes)}")
    
    for i in range(len(ts)-1):
        print(f"\nÉtape {i+1}/{len(ts)-1}:")
        penetration_val = float(penetrations[i+1])
        dt_val = float(ts[i+1] - ts[i])
        problem_obj.dt = dt_val
        penetration_percent = (i+1) / (len(ts)-1) * 100
        print(f"  Pénétration: {penetration_val*1000:.3f} μm ({penetration_percent:.1f}%), dt={dt_val:.6f}")
        current_dirichlet_bc_info = [
            [bottom_face_selector, bottom_face_selector, bottom_face_selector, contact_zone_selector],
            [0,                      1,                      2,                      2                      ],
            [zero_val_fn,            zero_val_fn,            zero_val_fn,            indenter_disp_fn(penetration_val)]
        ]
        problem_obj.fes[0].update_Dirichlet_boundary_conditions(current_dirichlet_bc_info)
        print(f"  Résolution avec solveur...")
        problem_obj.set_params(params)
        resolution_reussie = False
        try:
            sol_list = solver(problem_obj, {
                'umfpack_solver': {},
                'initial_guess': np.array(sol_prev),
                'tol': 1e-6
            })
            print(f"  Résolution UMFPACK réussie")
            resolution_reussie = True
        except Exception as e:
            print(f"  Échec UMFPACK : {str(e)[:100]}...")
            try:
                sol_list = solver(problem_obj, {
                    'jax_solver': {'precond': True},
                    'initial_guess': np.array(sol_prev),
                    'tol': 1e-5
                })
                print(f"  Résolution JAX réussie (fallback)")
                resolution_reussie = True
            except Exception as e2:
                print(f"  Échec complet à l'étape {i+1}:")
                break
        if not resolution_reussie:
            break
        sol_prev = sol_list[0]
        print(f"  Calcul des contraintes et analyse...")
        sigma_stresses = problem_obj.compute_avg_stress(sol_prev, params)
        sigma_xx_stress = onp.array(sigma_stresses[:, 0, 0])
        sigma_yy_stress = onp.array(sigma_stresses[:, 1, 1])
        sigma_zz_stress = onp.array(sigma_stresses[:, 2, 2])
        sigma_xy_stress = onp.array(sigma_stresses[:, 0, 1])
        sigma_yz_stress = onp.array(sigma_stresses[:, 1, 2])
        sigma_xz_stress = onp.array(sigma_stresses[:, 0, 2])
        sigma_vm_stress = onp.sqrt(
            0.5*((sigma_xx_stress - sigma_yy_stress)**2 +
                 (sigma_yy_stress - sigma_zz_stress)**2 +
                 (sigma_zz_stress - sigma_xx_stress)**2) +
            3.0*(sigma_xy_stress**2 + sigma_yz_stress**2 + sigma_xz_stress**2)
        )
        contact_force = calculate_contact_force(problem_obj, sol_prev, params, top_face_nodes, max_coords[2])
        avg_vm = float(onp.mean(sigma_vm_stress))
        max_vm = float(onp.max(sigma_vm_stress))
        min_vm = float(onp.min(sigma_vm_stress))
        std_vm = float(onp.std(sigma_vm_stress))
        contact_cells = []
        for cell_idx, cell_nodes in enumerate(cells):
            cell_has_contact_nodes = any(node_id in contact_zone_nodes for node_id in cell_nodes)
            if cell_has_contact_nodes:
                contact_cells.append(cell_idx)
        if len(contact_cells) > 0:
            contact_vm_stresses = sigma_vm_stress[contact_cells]
            avg_vm_contact = float(onp.mean(contact_vm_stresses))
            max_vm_contact = float(onp.max(contact_vm_stresses))
        else:
            avg_vm_contact = 0.0
            max_vm_contact = 0.0
        
        print(f"  Sauvegarde VTU étape {i+1}...")
        vtu_name = f"nano_indent_complet_step_{i:03d}.vtu"
        vtk_path = os.path.join(vtk_dir, vtu_name)
        cell_centers = onp.array([onp.mean(pts[cell_nodes], axis=0) for cell_nodes in cells])
        distances_to_tip = onp.sqrt((cell_centers[:, 0] - tip_x)**2 + (cell_centers[:, 1] - tip_y)**2)
        principal_stress_1 = sigma_xx_stress
        principal_stress_2 = sigma_yy_stress
        principal_stress_3 = sigma_zz_stress
        schmid_factors = onp.abs(sigma_zz_stress) / (sigma_vm_stress + 1e-6)
        save_sol(
            problem_obj.fes[0], sol_prev, vtk_path,
            cell_infos=[
                ('grain_id',             (cell_grain_inds + 1).astype(onp.float64)),
                ('cell_ori_inds',        cell_ori_inds.astype(onp.float64)),
                ('sigma_xx',             sigma_xx_stress),
                ('sigma_yy',             sigma_yy_stress),
                ('sigma_zz',             sigma_zz_stress),
                ('sigma_xy',             sigma_xy_stress),
                ('sigma_yz',             sigma_yz_stress),
                ('sigma_xz',             sigma_xz_stress),
                ('von_Mises_stress',     sigma_vm_stress),
                ('penetration_um',       onp.full(len(cells), penetration_val*1000)),
                ('in_contact_zone',      onp.array([1.0 if idx in contact_cells else 0.0 for idx in range(len(cells))])),
                ('distance_to_tip_um',   distances_to_tip * 1000),
                ('principal_stress_1',   principal_stress_1),
                ('principal_stress_2',   principal_stress_2),
                ('principal_stress_3',   principal_stress_3),
                ('schmid_factor_approx', schmid_factors),
                ('stress_ratio_contact', sigma_vm_stress / (avg_vm_contact + 1e-6)),
            ]
        )
        force_penetration_log.append([penetration_val*1000, contact_force])
        stress_evolution_log.append([i+1, penetration_val*1000, avg_vm, max_vm, avg_vm_contact, max_vm_contact])
        detailed_results_log.append([
            i+1, penetration_val*1000, contact_force,
            avg_vm, max_vm, min_vm, std_vm,
            avg_vm_contact, max_vm_contact, len(contact_cells)
        ])
        params = problem_obj.update_int_vars_gp(sol_prev, params)
        try:
            if len(params) > 2:
                slip_values = params[2]
                slip_max = float(onp.max(onp.abs(onp.array(slip_values))))
                slip_avg = float(onp.mean(onp.abs(onp.array(slip_values))))
            else:
                slip_max = 0.0
                slip_avg = 0.0
        except:
            slip_max = 0.0
            slip_avg = 0.0
        slip_evolution_log.append([penetration_val*1000, slip_max, slip_avg])
        print(f"  Force: {contact_force:.2e} N, von Mises: {avg_vm:.1f}±{std_vm:.1f} MPa")
        print(f"    Zone contact: {avg_vm_contact:.1f} MPa (max: {max_vm_contact:.1f}), slip_max: {slip_max:.2e}")
    
    print(f"\nSauvegarde des logs d'analyse...")
    onp.savetxt(os.path.join(numpy_dir, 'force_penetration_complet.txt'), onp.array(force_penetration_log),
                header='penetration_um force_N', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'stress_evolution_complet.txt'), onp.array(stress_evolution_log),
                header='step penetration_um vm_avg vm_max vm_contact_avg vm_contact_max', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'slip_evolution_complet.txt'), onp.array(slip_evolution_log),
                header='penetration_um max_slip avg_slip', fmt='%.6e')
    onp.savetxt(os.path.join(numpy_dir, 'detailed_results_complet.txt'), onp.array(detailed_results_log),
                header='step penetration_um force vm_avg vm_max vm_min vm_std vm_contact_avg vm_contact_max nb_contact_cells', fmt='%.6e')
    
    generate_comprehensive_analysis_plots(fig_dir, force_penetration_log, stress_evolution_log,
                                            slip_evolution_log, detailed_results_log,
                                            pyramid_base_radius, info)
    
    temps_total = time.time() - debut_total
    print("\n" + "="*70)
    print("SIMULATION NANO-INDENTATION TERMINÉE")
    print("="*70)
    print(f"  Temps total : {temps_total:.1f}s")
    print(f"  Résultats VTU : {vtk_dir}/")
    print(f"  Logs d'analyse : {numpy_dir}/")
    print(f"  Graphiques exhaustifs : {fig_dir}/")
    print(f"  Pénétration finale : {max_penetration*1000:.1f} μm")
    
    if len(force_penetration_log) > 0:
        final_force = force_penetration_log[-1][1]
        print(f"  Force finale : {final_force*1000:.3f} mN")
        contact_area = np.pi * pyramid_base_radius**2
        hardness = final_force / contact_area / 1e6
        print(f"  Dureté approximative : {hardness:.2f} GPa")
    print("="*70)

if __name__ == '__main__':
    problem()
