import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from skimage import measure, morphology
from sklearn.cluster import DBSCAN
import gmsh
import meshio
import argparse
import time

def setup_directories():
    base_dir = os.path.dirname(__file__)
    if not base_dir:
        base_dir = "."
    data_dir = os.path.join(base_dir, 'data')
    directories = {
        'debug_visualizations': os.path.join(data_dir, 'debug_visualizations'),
        'ctf': os.path.join(data_dir, 'ctf'),
        'neper': os.path.join(data_dir, 'neper', 'traction_cuivre'),
        'csv': os.path.join(data_dir, 'csv', 'traction_cuivre'),
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    print(f"Dossier de visualisations : {directories['debug_visualizations']}")
    return directories

def find_ctf_file(ctf_dir):
    ctf_files = [f for f in os.listdir(ctf_dir) if f.endswith('.ctf')]
    if not ctf_files:
        raise FileNotFoundError(f"Aucun fichier .ctf trouvé dans {ctf_dir}")
    if len(ctf_files) > 1:
        print(f"Plusieurs fichiers .ctf trouvés. Utilisation du premier : {ctf_files[0]}")
    ctf_path = os.path.join(ctf_dir, ctf_files[0])
    print(f"Fichier CTF trouvé : {ctf_path}")
    return ctf_path

def analyze_ctf_structure(ctf_path):
    print(f"Analyse de la structure du fichier CTF...")
    with open(ctf_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    print(f"Nombre total de lignes : {len(lines)}")
    xcells, ycells, xstep, ystep = None, None, None, None
    for line in lines[:20]:
        if line.startswith('XCells'):
            xcells = int(line.split('\t')[1])
        elif line.startswith('YCells'):
            ycells = int(line.split('\t')[1])
        elif line.startswith('XStep'):
            xstep = float(line.split('\t')[1])
        elif line.startswith('YStep'):
            ystep = float(line.split('\t')[1])
    print(f"En-têtes CTF détectés : XCells={xcells}, YCells={ycells}, XStep={xstep}, YStep={ystep}")
    
    data_start_candidates = []
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if line_clean and not line_clean.startswith('#') and not line_clean.startswith('Channel'):
            parts = line_clean.split()
            if len(parts) >= 8:
                try:
                    int(parts[0])
                    float(parts[1])
                    float(parts[2])
                    float(parts[5])
                    data_start_candidates.append(i)
                except ValueError:
                    continue
    if not data_start_candidates:
        raise ValueError("Aucune ligne de données numériques trouvée")
    data_start = data_start_candidates[0]
    print(f"Début des données détecté à la ligne : {data_start}")
    sample_line = lines[data_start].strip().split()
    print(f"Nombre de colonnes détecté : {len(sample_line)}")
    if xcells and ycells:
        expected_data_points = xcells * ycells
        available_data_lines = len(lines) - data_start
        print(f"Points attendus d'après en-têtes : {expected_data_points}")
        print(f"Lignes de données disponibles : {available_data_lines}")
        if available_data_lines < expected_data_points * 0.9:
            print(f"[WARNING] Nombre de lignes insuffisant par rapport aux en-têtes")
    return data_start, lines, (xcells, ycells, xstep, ystep)

def read_ctf_file(ctf_path):
    print(f"\nLecture et Segmentation Initiale...")
    try:
        data_start, lines, header_info = analyze_ctf_structure(ctf_path)
        xcells, ycells, xstep, ystep = header_info
        print(f"Lecture des données à partir de la ligne {data_start}...")
        data_rows = []
        valid_lines = 0
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                phase = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                euler1 = float(parts[5])
                euler2 = float(parts[6])
                euler3 = float(parts[7])
                data_rows.append([x, y, euler1, euler2, euler3])
                valid_lines += 1
            except (ValueError, IndexError) as e:
                continue
        print(f"{valid_lines} lignes de données valides extraites")
        if valid_lines == 0:
            raise ValueError("Aucune donnée valide extraite du fichier CTF")
        if xcells and ycells:
            expected_points = xcells * ycells
            if abs(valid_lines - expected_points) > expected_points * 0.05:
                print(f"[WARNING] Différence significative entre points lus et attendus")
        
        data_array = np.array(data_rows)
        x_coords = data_array[:, 0]
        y_coords = data_array[:, 1]
        euler1 = data_array[:, 2]
        euler2 = data_array[:, 3]
        euler3 = data_array[:, 4]
        
        unique_x = np.unique(x_coords)
        unique_y = np.unique(y_coords)
        nx = len(unique_x)
        ny = len(unique_y)
        if nx <= 1 or ny <= 1:
            raise ValueError(f"Dimensions insuffisantes: nx={nx}, ny={ny}")
        step_x = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 1.0
        step_y = unique_y[1] - unique_y[0] if len(unique_y) > 1 else 1.0
        print(f"Dimensions EBSD : {nx} × {ny} pixels")
        
        if xcells and ycells and xstep and ystep:
            physical_width_header = (xcells - 1) * xstep
            physical_height_header = (ycells - 1) * ystep
            if abs((nx - 1) * step_x - physical_width_header) < physical_width_header * 0.1:
                physical_width = physical_width_header
                physical_height = physical_height_header
            else:
                physical_width = (nx - 1) * step_x
                physical_height = (ny - 1) * step_y
        else:
            physical_width = (nx - 1) * step_x
            physical_height = (ny - 1) * step_y
        
        print(f"Dimensions physiques finales : {physical_width:.3f} × {physical_height:.3f}")
        
        euler1_grid = np.zeros((ny, nx))
        euler2_grid = np.zeros((ny, nx))
        euler3_grid = np.zeros((ny, nx))
        points_mapped = 0
        for i in range(len(x_coords)):
            x_val = x_coords[i]
            y_val = y_coords[i]
            x_idx = np.argmin(np.abs(unique_x - x_val))
            y_idx = np.argmin(np.abs(unique_y - y_val))
            if (np.abs(unique_x[x_idx] - x_val) < step_x * 0.1 and
                np.abs(unique_y[y_idx] - y_val) < step_y * 0.1):
                euler1_grid[y_idx, x_idx] = euler1[i]
                euler2_grid[y_idx, x_idx] = euler2[i]
                euler3_grid[y_idx, x_idx] = euler3[i]
                points_mapped += 1
        print(f"{points_mapped}/{len(x_coords)} points mappés sur la grille")
        
        return {
            'euler1': euler1_grid, 'euler2': euler2_grid, 'euler3': euler3_grid,
            'nx': nx, 'ny': ny, 'step_x': step_x, 'step_y': step_y,
            'physical_width': physical_width, 'physical_height': physical_height
        }
    except Exception as e:
        print(f"[ERREUR] Lecture du fichier CTF : {e}")
        raise

def euler_to_quaternion(euler1, euler2, euler3):
    phi1 = np.radians(euler1)
    Phi = np.radians(euler2)  
    phi2 = np.radians(euler3)
    q0 = np.cos(Phi/2) * np.cos((phi1 + phi2)/2)
    q1 = np.sin(Phi/2) * np.cos((phi1 - phi2)/2)
    q2 = np.sin(Phi/2) * np.sin((phi1 - phi2)/2)
    q3 = np.cos(Phi/2) * np.sin((phi1 + phi2)/2)
    return np.column_stack([q0, q1, q2, q3])

def create_ipf_colors(euler1, euler2, euler3):
    quaternions = euler_to_quaternion(euler1.flatten(), euler2.flatten(), euler3.flatten())
    colors = np.abs(quaternions[:, 1:4])
    colors = colors / np.max(colors, axis=1, keepdims=True)
    return colors.reshape(euler1.shape + (3,))

def save_debug_image(data, title, filename, debug_dir, grain_ids=None):
    plt.figure(figsize=(12, 9))
    if grain_ids is not None:
        plt.imshow(grain_ids, cmap='tab20', origin='lower')
        plt.title(f"{title} - {len(np.unique(grain_ids))} grains")
        plt.colorbar(label="Grain ID")
    else:
        plt.imshow(data, origin='lower')
        plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    filepath = os.path.join(debug_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Image sauvegardée : {filename}")

def segment_grains_advanced(ebsd_data, debug_dir):
    euler1 = ebsd_data['euler1']
    euler2 = ebsd_data['euler2']
    euler3 = ebsd_data['euler3']
    ipf_colors = create_ipf_colors(euler1, euler2, euler3)
    save_debug_image(ipf_colors, "Orientations EBSD brutes (IPF)", "00_orientations_brutes.png", debug_dir)
    print(f"Segmentation initiale des grains...")
    ny, nx = euler1.shape
    grad_euler1_x = np.abs(np.gradient(euler1, axis=1))
    grad_euler1_y = np.abs(np.gradient(euler1, axis=0))
    grad_euler2_x = np.abs(np.gradient(euler2, axis=1))
    grad_euler2_y = np.abs(np.gradient(euler2, axis=0))
    grad_euler3_x = np.abs(np.gradient(euler3, axis=1))
    grad_euler3_y = np.abs(np.gradient(euler3, axis=0))
    grad_magnitude = np.sqrt(
        grad_euler1_x**2 + grad_euler1_y**2 +
        grad_euler2_x**2 + grad_euler2_y**2 +
        grad_euler3_x**2 + grad_euler3_y**2
    )
    threshold = np.percentile(grad_magnitude, 80)
    boundaries = grad_magnitude > threshold
    from scipy.ndimage import label
    from skimage.segmentation import watershed
    markers = ~boundaries
    markers = morphology.remove_small_objects(markers, min_size=50)
    markers = label(markers)[0]
    grain_ids = watershed(grad_magnitude, markers, mask=~boundaries)
    print(f"-> {len(np.unique(grain_ids))} grains initiaux détectés.")
    print(f"Nettoyage de la carte des grains...")
    min_grain_size = max(50, (nx * ny) // 10000)
    unique_ids, counts = np.unique(grain_ids, return_counts=True)
    large_grains_mask = np.isin(grain_ids, unique_ids[counts >= min_grain_size])
    grain_ids_filtered = grain_ids * large_grains_mask
    print(f"-> {len(np.unique(grain_ids_filtered))-1} grains après filtrage par taille.")
    print(f"-> Trous intra-granulaires remplis.")
    grain_ids_filled = morphology.remove_small_holes(grain_ids_filtered > 0, area_threshold=100)
    grain_ids_final = grain_ids_filtered.copy()
    grain_ids_final[grain_ids_filtered == 0] = watershed(
        grad_magnitude, grain_ids_filtered, mask=grain_ids_filled
    )[grain_ids_filtered == 0]
    save_debug_image(ipf_colors, "Segmentation initiale", "01_segmentation_initiale.png", debug_dir, grain_ids_final)
    return grain_ids_final, ipf_colors

def clean_grain_map(grain_ids, ipf_colors, debug_dir):
    print(f"\nNettoyage Geometrique ({len(np.unique(grain_ids))} grains initiaux)")
    save_debug_image(ipf_colors, "Après filtrage par taille", "02_apres_filtrage_taille.png", debug_dir, grain_ids)
    save_debug_image(ipf_colors, "Après remplissage des trous", "03_apres_remplissage_trous.png", debug_dir, grain_ids)
    return grain_ids

def expand_grains_to_fill_voids(grain_ids, ipf_colors, debug_dir):
    print(f"\nExpansion des Grains pour Combler les Vides")
    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed
    voids = (grain_ids == 0)
    if not np.any(voids):
        print(f"Aucun vide détecté.")
    else:
        distance = distance_transform_edt(voids)
        expanded_grains = watershed(-distance, grain_ids, mask=np.ones_like(grain_ids, dtype=bool))
        grain_ids = expanded_grains
        print(f"Tous les vides ont été comblés par expansion des grains voisins.")
    save_debug_image(ipf_colors, "Après expansion des grains", "04_apres_expansion_des_grains.png", debug_dir, grain_ids)
    return grain_ids

def renumber_grains_consecutively(grain_ids, ipf_colors, debug_dir):
    print(f"\nRenommage des Grains en IDs Consecutifs")
    unique_ids = np.unique(grain_ids)
    unique_ids = unique_ids[unique_ids > 0]
    new_grain_ids = np.zeros_like(grain_ids)
    for new_id, old_id in enumerate(unique_ids, start=1):
        new_grain_ids[grain_ids == old_id] = new_id
    print(f"-> Renommage terminé : {len(unique_ids)} grains -> IDs de 1 à {len(unique_ids)}")
    save_debug_image(ipf_colors, "Après renommage des grains", "05_apres_renommage_grains.png", debug_dir, new_grain_ids)
    return new_grain_ids

def compute_grain_orientations(grain_ids, euler1, euler2, euler3):
    print(f"\nCalcul des orientations moyennes par grain")
    unique_grains = np.unique(grain_ids)
    unique_grains = unique_grains[unique_grains > 0]
    grain_orientations = []
    for grain_id in unique_grains:
        mask = (grain_ids == grain_id)
        mean_euler1 = np.mean(euler1[mask])
        mean_euler2 = np.mean(euler2[mask])
        mean_euler3 = np.mean(euler3[mask])
        grain_orientations.append([grain_id, mean_euler1, mean_euler2, mean_euler3])
    grain_orientations = np.array(grain_orientations)
    print(f"-> {len(grain_orientations)} grains avec orientation moyenne calculée.")
    return grain_orientations

def create_structured_mesh_reduced_resolution(grain_ids, ebsd_data, debug_dir):
    print(f"\nCreation du Maillage Structure")
    ny, nx = grain_ids.shape
    if nx < 20 or ny < 20:
        element_size_x = max(1, nx // 4)
        element_size_y = max(1, ny // 100)
    else:
        element_size_x = 20
        element_size_y = 20
    new_nx = max(1, nx // element_size_x)
    new_ny = max(1, ny // element_size_y)
    print(f"Nouvelle résolution : {new_nx} × {new_ny} éléments")
    mesh_grain_ids = np.zeros((new_ny, new_nx), dtype=int)
    for j in range(new_ny):
        for i in range(new_nx):
            y_start, y_end = j * element_size_y, min((j + 1) * element_size_y, ny)
            x_start, x_end = i * element_size_x, min((i + 1) * element_size_x, nx)
            region = grain_ids[y_start:y_end, x_start:x_end]
            unique_vals, counts = np.unique(region, return_counts=True)
            mesh_grain_ids[j, i] = unique_vals[np.argmax(counts)]
    final_grains = np.unique(mesh_grain_ids)
    final_grains = final_grains[final_grains > 0]
    if len(final_grains) == 0:
        raise ValueError("Maillage vide - aucun grain détecté")
    save_debug_image(None, "Maillage structuré final", "06_maillage_final_structure.png", debug_dir, mesh_grain_ids)
    return {
        'grain_ids': mesh_grain_ids, 'nx': new_nx, 'ny': new_ny,
        'physical_width': ebsd_data['physical_width'], 'physical_height': ebsd_data['physical_height'],
        'thickness': 20.0, 'final_grains': final_grains
    }

def export_mesh_and_quaternions(mesh_data, grain_orientations, directories):
    print(f"\nExport des fichiers")
    grain_ids = mesh_data['grain_ids']
    nx, ny = mesh_data['nx'], mesh_data['ny']
    width, height, thickness = mesh_data['physical_width'], mesh_data['physical_height'], mesh_data['thickness']
    final_grains = mesh_data['final_grains']
    
    grain_orientations_filtered = []
    for grain_data in grain_orientations:
        if int(grain_data[0]) in final_grains:
            grain_orientations_filtered.append(grain_data)
    
    msh_path = os.path.join(directories['neper'], 'domain.msh')
    print(f"Export du maillage au format .msh : '{msh_path}'...")
    
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("domain")
    try:
        nz = 6
        if nx == 0 or ny == 0: raise ValueError(f"Dimensions de maillage nulles: nx={nx}, ny={ny}")
        element_width, element_height, element_thickness = width / nx, height / ny, thickness / nz
        
        nodes, node_map = [], {}
        node_id = 0
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    nodes.append([i * element_width, j * element_height, k * element_thickness])
                    node_map[(i, j, k)] = node_id
                    node_id += 1
        
        hex_elements = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    n = [node_map[(i,j,k)], node_map[(i+1,j,k)], node_map[(i+1,j+1,k)], node_map[(i,j+1,k)],
                         node_map[(i,j,k+1)], node_map[(i+1,j,k+1)], node_map[(i+1,j+1,k+1)], node_map[(i,j+1,k+1)]]
                    hex_elements.append(n)
        
        mesh_meshio = meshio.Mesh(np.array(nodes), [("hexahedron", np.array(hex_elements))])
        mesh_meshio.cell_data = {"gmsh:physical": [grain_ids.flatten('F').repeat(nz)]}
        meshio.write(msh_path, mesh_meshio, file_format='gmsh22', binary=False)
        print(f"Export meshio réussi : {len(hex_elements)} éléments hexaédriques")

    finally:
        gmsh.finalize()

    vtk_path = os.path.join(directories['debug_visualizations'], 'maillage_final_structure.vtk')
    try:
        meshio.write(vtk_path, meshio.read(msh_path), file_format='vtk')
    except Exception as e:
        print(f"[ERREUR] Export VTK échoué : {e}")

    quat_path = os.path.join(directories['csv'], 'quat.txt')
    print(f"Export des quaternions : '{quat_path}'...")
    if len(final_grains) == 0:
        np.savetxt(quat_path, np.array([]))
        return {'nb_elements': 0, 'nb_grains_final': 0, 'physical_dimensions': (width, height, thickness)}

    quaternions_export = []
    grain_id_mapping = {grain_id: idx for idx, grain_id in enumerate(sorted(final_grains))}
    for grain_data in grain_orientations_filtered:
        grain_id = int(grain_data[0])
        if grain_id in final_grains:
            quat = euler_to_quaternion(np.array([grain_data[1]]), np.array([grain_data[2]]), np.array([grain_data[3]]))[0]
            quaternions_export.append([grain_id_mapping[grain_id], quat[0], quat[1], quat[2], quat[3]])
    quaternions_export.sort(key=lambda x: x[0])
    np.savetxt(quat_path, np.array(quaternions_export), fmt='%d %.6f %.6f %.6f %.6f', header='Index Q0 Q1 Q2 Q3')
    print(f"{len(quaternions_export)} quaternions exportés.")
    
    return {
        'nb_elements': nx * ny * nz,
        'nb_grains_final': len(final_grains),
        'physical_dimensions': (width, height, thickness)
    }

def main():
    start_time = time.time()
    directories = setup_directories()
    ctf_path = find_ctf_file(directories['ctf'])
    ebsd_data = read_ctf_file(ctf_path)
    grain_ids, ipf_colors = segment_grains_advanced(ebsd_data, directories['debug_visualizations'])
    grain_ids = clean_grain_map(grain_ids, ipf_colors, directories['debug_visualizations'])
    grain_ids = expand_grains_to_fill_voids(grain_ids, ipf_colors, directories['debug_visualizations'])
    grain_ids = renumber_grains_consecutively(grain_ids, ipf_colors, directories['debug_visualizations'])
    grain_orientations = compute_grain_orientations(grain_ids, ebsd_data['euler1'], ebsd_data['euler2'], ebsd_data['euler3'])
    mesh_data = create_structured_mesh_reduced_resolution(grain_ids, ebsd_data, directories['debug_visualizations'])
    export_results = export_mesh_and_quaternions(mesh_data, grain_orientations, directories)
    execution_time = time.time() - start_time
    print(f"\nTemps d'exécution : {execution_time:.1f}s")

if __name__ == "__main__":
    main()
