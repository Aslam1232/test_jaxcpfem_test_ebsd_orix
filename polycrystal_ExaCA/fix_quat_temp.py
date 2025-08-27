"""
Synchronisation quaternions avec maillage grossier
Après création du maillage grossier, certains grains disparaissent
Il faut adapter le fichier quaternions pour correspondre exactement
"""

import os
import numpy as np
import meshio
import shutil
from datetime import datetime

def analyser_grains_maillage_grossier():
    print("="*70)
    print("ANALYSE GRAINS CONSERVÉS DANS MAILLAGE GROSSIER")
    print("="*70)
    
    msh_file = "data/msh_with_grains/exaca_grains_only.msh"
    
    if not os.path.exists(msh_file):
        print(f"ERREUR: Maillage grossier non trouvé: {msh_file}")
        return None
    
    print(f"Lecture maillage grossier: {msh_file}")
    
    mesh = meshio.read(msh_file)
    
    if 'gmsh:physical' not in mesh.cell_data:
        print("ERREUR: Pas de physical groups dans le maillage")
        return None
    
    physical_groups = mesh.cell_data['gmsh:physical'][0]
    unique_physical_groups = np.unique(physical_groups)
    
    print(f"Points dans maillage grossier: {len(mesh.points):,}")
    print(f"Cellules dans maillage grossier: {len(mesh.cells_dict['hexahedron']):,}")
    print(f"Physical groups uniques: {len(unique_physical_groups)}")
    print(f"Range physical groups: {unique_physical_groups[0]} à {unique_physical_groups[-1]}")
    
    return {
        'mesh': mesh,
        'physical_groups': physical_groups,
        'unique_physical_groups': unique_physical_groups,
        'n_cells': len(mesh.cells_dict['hexahedron']),
        'n_points': len(mesh.points)
    }

def analyser_quaternions_originaux():
    print("\n" + "="*70)
    print("ANALYSE QUATERNIONS ORIGINAUX")
    print("="*70)
    
    quat_file = "data/csv/polycrystal_ExaCA/quat_final.txt"
    
    if not os.path.exists(quat_file):
        print(f"ERREUR: Fichier quaternions non trouvé: {quat_file}")
        return None
    
    print(f"Lecture quaternions: {quat_file}")
    
    quat_data = np.loadtxt(quat_file)
    if quat_data.ndim == 1:
        quat_data = quat_data.reshape(1, -1)
    
    indices_quat = quat_data[:, 0].astype(int)
    quaternions = quat_data[:, 1:]
    
    print(f"Quaternions chargés: {len(quaternions)}")
    print(f"Indices quaternions: {indices_quat[0]} à {indices_quat[-1]}")
    print(f"Shape quaternions: {quaternions.shape}")
    
    return {
        'quat_data': quat_data,
        'indices_quat': indices_quat,
        'quaternions': quaternions,
        'n_quaternions': len(quaternions)
    }

def creer_mapping_grains_conserves(mesh_info, quat_info):
    print("\n" + "="*70)
    print("CRÉATION MAPPING GRAINS CONSERVÉS")
    print("="*70)
    
    conserved_physical_groups = mesh_info['unique_physical_groups']
    
    needed_quat_indices = conserved_physical_groups - 1
    
    max_needed_index = np.max(needed_quat_indices)
    available_quat_indices = quat_info['indices_quat']
    max_available_index = np.max(available_quat_indices)
    
    print(f"Grains conservés (physical groups): {len(conserved_physical_groups)}")
    print(f"Range physical groups: {conserved_physical_groups[0]} à {conserved_physical_groups[-1]}")
    print(f"Indices quaternions nécessaires: {needed_quat_indices[0]} à {needed_quat_indices[-1]}")
    print(f"Indices quaternions disponibles: {available_quat_indices[0]} à {max_available_index}")
    
    if max_needed_index > max_available_index:
        print(f"ERREUR: Quaternion {max_needed_index} nécessaire mais max disponible = {max_available_index}")
        return None
    
    missing_quaternions = []
    for needed_idx in needed_quat_indices:
        if needed_idx not in available_quat_indices:
            missing_quaternions.append(needed_idx)
    
    if missing_quaternions:
        print(f"ERREUR: Quaternions manquants: {missing_quaternions}")
        return None
    
    print("OK: Tous les quaternions nécessaires sont disponibles")
    
    return {
        'conserved_physical_groups': conserved_physical_groups,
        'needed_quat_indices': needed_quat_indices,
        'n_conserved_grains': len(conserved_physical_groups)
    }

def creer_fichiers_synchronises(mesh_info, quat_info, mapping_info):
    print("\n" + "="*70)
    print("CRÉATION FICHIERS SYNCHRONISÉS")
    print("="*70)
    
    conserved_physical_groups = mapping_info['conserved_physical_groups']
    needed_quat_indices = mapping_info['needed_quat_indices']
    
    conserved_quaternions = []
    for i, needed_idx in enumerate(needed_quat_indices):
        quat_pos = np.where(quat_info['indices_quat'] == needed_idx)[0]
        if len(quat_pos) > 0:
            quat_pos = quat_pos[0]
            quaternion = quat_info['quaternions'][quat_pos]
            conserved_quaternions.append([i, quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    
    conserved_quaternions = np.array(conserved_quaternions)
    
    print(f"Quaternions conservés: {len(conserved_quaternions)}")
    
    quat_dir = "data/csv/polycrystal_ExaCA"
    os.makedirs(quat_dir, exist_ok=True)
    
    old_quat_file = os.path.join(quat_dir, "quat_final.txt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_quat_file = os.path.join(quat_dir, f"quat_final_backup_{timestamp}.txt")
    
    if os.path.exists(old_quat_file):
        shutil.copy2(old_quat_file, backup_quat_file)
        print(f"Backup quaternions original: {backup_quat_file}")
    
    new_quat_file = os.path.join(quat_dir, "quat_final.txt")
    np.savetxt(new_quat_file, conserved_quaternions, fmt='%d %.15f %.15f %.15f %.15f')
    print(f"Nouveau fichier quaternions: {new_quat_file}")
    
    old_physical_groups = mesh_info['physical_groups']
    new_physical_groups = np.zeros_like(old_physical_groups)
    
    old_to_new_mapping = {}
    for new_idx, old_physical_group in enumerate(conserved_physical_groups):
        old_to_new_mapping[old_physical_group] = new_idx + 1
    
    for i, old_pg in enumerate(old_physical_groups):
        new_physical_groups[i] = old_to_new_mapping[old_pg]
    
    print(f"Physical groups renumérotés: {np.unique(old_physical_groups)} → {np.unique(new_physical_groups)}")
    
    new_mesh = meshio.Mesh(
        points=mesh_info['mesh'].points,
        cells=mesh_info['mesh'].cells
    )
    
    new_mesh.cell_data = {
        "gmsh:physical": [new_physical_groups]
    }
    
    if hasattr(mesh_info['mesh'], 'field_data'):
        new_mesh.field_data = {}
        for new_idx, old_physical_group in enumerate(conserved_physical_groups):
            new_physical_id = new_idx + 1
            grain_name = f"Grain_{old_physical_group}"
            new_mesh.field_data[grain_name] = np.array([new_physical_id, 3])
    
    msh_dir = "data/msh_with_grains"
    old_msh_file = os.path.join(msh_dir, "exaca_grains_only.msh")
    backup_msh_file = os.path.join(msh_dir, f"exaca_grains_only_backup_{timestamp}.msh")
    
    if os.path.exists(old_msh_file):
        shutil.copy2(old_msh_file, backup_msh_file)
        print(f"Backup maillage grossier: {backup_msh_file}")
    
    new_msh_file = os.path.join(msh_dir, "exaca_grains_only.msh")
    meshio.write(new_msh_file, new_mesh, file_format='gmsh22')
    print(f"Nouveau maillage synchronisé: {new_msh_file}")
    
    return {
        'new_quat_file': new_quat_file,
        'new_msh_file': new_msh_file,
        'n_conserved_grains': len(conserved_quaternions),
        'old_to_new_mapping': old_to_new_mapping
    }

def main():
    print("SYNCHRONISATION MAILLAGE GROSSIER ↔ QUATERNIONS")
    print("="*70)
    
    mesh_info = analyser_grains_maillage_grossier()
    if mesh_info is None:
        return
    
    quat_info = analyser_quaternions_originaux()
    if quat_info is None:
        return
    
    mapping_info = creer_mapping_grains_conserves(mesh_info, quat_info)
    if mapping_info is None:
        return
    
    sync_info = creer_fichiers_synchronises(mesh_info, quat_info, mapping_info)
    if sync_info is None:
        return
    
    print("\n" + "="*70)
    print("SYNCHRONISATION TERMINÉE")
    print("="*70)
    print(f"Grains conservés: {sync_info['n_conserved_grains']}/169")
    print(f"Cellules maillage grossier: {mesh_info['n_cells']:,}")
    print(f"Points maillage grossier: {mesh_info['n_points']:,}")
    print(f"Nouveau fichier quaternions: {sync_info['n_conserved_grains']} orientations")
    print(f"Physical groups: renumérotés 1 à {sync_info['n_conserved_grains']}")
    print(f"Indices quaternions: renumérotés 0 à {sync_info['n_conserved_grains']-1}")
    print("")
    print("Fichiers prêts pour la simulation.")
    print(f"Maillage: {sync_info['new_msh_file']}")
    print(f"Quaternions: {sync_info['new_quat_file']}")
    print("="*70)

if __name__ == "__main__":
    main()
