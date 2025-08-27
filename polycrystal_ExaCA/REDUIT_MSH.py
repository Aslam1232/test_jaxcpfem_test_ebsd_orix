import vtk
import numpy as np
import os
from vtk.util.numpy_support import vtk_to_numpy

def create_coarse_msh_from_vtk():
    print("="*70)
    print("CRÉATION MAILLAGE GROSSIER DEPUIS VTK EXACA")
    print("Regroupement de voxels en hexaèdres plus gros")
    print("="*70)

    vtk_file = './Aslam/Test-Inp_SingleLineBinary/TestProblemSingleLine.vtk'
    
    if not os.path.exists(vtk_file):
        print(f"ERREUR: Fichier VTK non trouvé: {vtk_file}")
        return None
    
    print(f"Lecture VTK original : {os.path.basename(vtk_file)}")
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    data = reader.GetOutput()
    
    dims = data.GetDimensions()
    spacing = data.GetSpacing()
    origin = data.GetOrigin()
    
    points = data.GetPoints()
    numpy_points = vtk_to_numpy(points.GetData())
    
    grain_array = data.GetPointData().GetArray('GrainID')
    grain_data = vtk_to_numpy(grain_array)
    
    nx, ny, nz = dims
    print(f"Dimensions VTK originales : {nx} × {ny} × {nz} = {nx*ny*nz:,} points")
    print(f"Espacement : {spacing}")
    print(f"Origine : {origin}")
    print(f"GrainIDs : range [{grain_data.min()}, {grain_data.max()}]")
    
    coarse_factor = 2 
    
    print(f"\nFacteur de regroupement : {coarse_factor}")
    print(f"Chaque hexaèdre représentera un bloc {coarse_factor}×{coarse_factor}×{coarse_factor}")
    
    nx_coarse = (nx - 1) // coarse_factor
    ny_coarse = (ny - 1) // coarse_factor  
    nz_coarse = (nz - 1) // coarse_factor
    
    print(f"Nouvelles dimensions : {nx_coarse} × {ny_coarse} × {nz_coarse}")
    print(f"Nouvelles cellules estimées : {(nx_coarse-1)*(ny_coarse-1)*(nz_coarse-1):,}")
    
    if nx_coarse < 2 or ny_coarse < 2 or nz_coarse < 2:
        print(f"ERREUR: Facteur de regroupement trop grand. Réduisez coarse_factor.")
        return None
    
    new_points = []
    
    for k in range(nz_coarse):
        for j in range(ny_coarse):
            for i in range(nx_coarse):
                orig_i = i * coarse_factor
                orig_j = j * coarse_factor
                orig_k = k * coarse_factor
                
                x = origin[0] + orig_i * spacing[0]
                y = origin[1] + orig_j * spacing[1]
                z = origin[2] + orig_k * spacing[2]
                
                new_points.append([x, y, z])
    
    new_points = np.array(new_points)
    print(f"Nouveaux points créés : {len(new_points):,}")
    
    coarse_cells = []
    coarse_grain_ids = []
    
    for k in range(nz_coarse - 1):
        for j in range(ny_coarse - 1):
            for i in range(nx_coarse - 1):
                i000 = k * nx_coarse * ny_coarse + j * nx_coarse + i
                i100 = k * nx_coarse * ny_coarse + j * nx_coarse + (i+1)
                i010 = k * nx_coarse * ny_coarse + (j+1) * nx_coarse + i
                i110 = k * nx_coarse * ny_coarse + (j+1) * nx_coarse + (i+1)
                i001 = (k+1) * nx_coarse * ny_coarse + j * nx_coarse + i
                i101 = (k+1) * nx_coarse * ny_coarse + j * nx_coarse + (i+1)
                i011 = (k+1) * nx_coarse * ny_coarse + (j+1) * nx_coarse + i
                i111 = (k+1) * nx_coarse * ny_coarse + (j+1) * nx_coarse + (i+1)
                
                cell = [i000+1, i100+1, i110+1, i010+1, i001+1, i101+1, i111+1, i011+1]
                coarse_cells.append(cell)
                
                sample_grains = []
                
                for dk in range(coarse_factor):
                    for dj in range(coarse_factor):
                        for di in range(coarse_factor):
                            orig_i = i * coarse_factor + di
                            orig_j = j * coarse_factor + dj
                            orig_k = k * coarse_factor + dk
                            
                            if orig_i < nx and orig_j < ny and orig_k < nz:
                                orig_idx = orig_k * nx * ny + orig_j * nx + orig_i
                                if orig_idx < len(grain_data):
                                    grain_id = grain_data[orig_idx]
                                    if grain_id >= 0:
                                        sample_grains.append(grain_id)
                
                if sample_grains:
                    unique_grains, counts = np.unique(sample_grains, return_counts=True)
                    majority_grain = unique_grains[np.argmax(counts)]
                    coarse_grain_ids.append(majority_grain)
                else:
                    coarse_grain_ids.append(1)
    
    coarse_grain_ids = np.array(coarse_grain_ids)
    
    print(f"Cellules grossières créées : {len(coarse_cells):,}")
    
    unique_coarse_grains = np.unique(coarse_grain_ids)
    valid_coarse_grains = unique_coarse_grains[unique_coarse_grains >= 0]
    
    print(f"Grains dans maillage grossier : {len(valid_coarse_grains)}")
    print(f"Range grains : {valid_coarse_grains[0]} à {valid_coarse_grains[-1]}")
    
    grain_to_physical = {}
    for i, grain_id in enumerate(valid_coarse_grains):
        grain_to_physical[grain_id] = i + 1
    
    output_dir = "data/msh_with_grains"
    os.makedirs(output_dir, exist_ok=True)
    msh_file = os.path.join(output_dir, "exaca_grains_only.msh")
    
    print(f"Écriture fichier MSH grossier : {msh_file}")
    
    with open(msh_file, 'w') as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        f.write("$Nodes\n")
        f.write(f"{len(new_points)}\n")
        
        for i, point in enumerate(new_points):
            f.write(f"{i+1} {point[0]:.10e} {point[1]:.10e} {point[2]:.10e}\n")
        
        f.write("$EndNodes\n")
        
        f.write("$Elements\n")
        f.write(f"{len(coarse_cells)}\n")
        
        for i, cell in enumerate(coarse_cells):
            grain_id = coarse_grain_ids[i]
            physical_group = grain_to_physical.get(grain_id, 1)
            
            node_list = ' '.join(map(str, cell))
            f.write(f"{i+1} 5 2 {physical_group} {physical_group} {node_list}\n")
        
        f.write("$EndElements\n")
        
        f.write("$PhysicalNames\n")
        f.write(f"{len(valid_coarse_grains)}\n")
        
        for grain_id in valid_coarse_grains:
            physical_id = grain_to_physical[grain_id]
            f.write(f'3 {physical_id} "Grain_{grain_id}"\n')
        
        f.write("$EndPhysicalNames\n")
    
    min_coords_new = np.min(new_points, axis=0)
    max_coords_new = np.max(new_points, axis=0)
    dimensions_new = max_coords_new - min_coords_new
    
    print(f"\n" + "="*70)
    print("MAILLAGE GROSSIER CRÉÉ AVEC SUCCÈS")
    print("="*70)
    print(f"Fichier MSH : {msh_file}")
    print(f"Points : {len(new_points):,}")
    print(f"Cellules : {len(coarse_cells):,}")
    print(f"Grains : {len(valid_coarse_grains)}")
    print(f"Facteur de réduction : {(nx-1)*(ny-1)*(nz-1) / len(coarse_cells):.1f}x")
    print(f"Dimensions : {dimensions_new[0]*1000:.1f} × {dimensions_new[1]*1000:.1f} × {dimensions_new[2]*1000:.1f} mm")
    print(f"Coordonnées min : {min_coords_new}")
    print(f"Coordonnées max : {max_coords_new}")
    print("="*70)
    
    return {
        'msh_file': msh_file,
        'n_points': len(new_points),
        'n_cells': len(coarse_cells),
        'n_grains': len(valid_coarse_grains),
        'dimensions': dimensions_new,
        'reduction_factor': (nx-1)*(ny-1)*(nz-1) / len(coarse_cells)
    }

def test_different_coarse_factors():
    print("="*70)
    print("TEST DE DIFFÉRENTS FACTEURS DE REGROUPEMENT")
    print("="*70)
    
    vtk_file = './Aslam/Test-Inp_SingleLineBinary/TestProblemSingleLine.vtk'
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    data = reader.GetOutput()
    dims = data.GetDimensions()
    nx, ny, nz = dims
    
    original_cells = (nx-1) * (ny-1) * (nz-1)
    
    print(f"Cellules originales : {original_cells:,}")
    print("")
    
    for factor in [2, 3, 4, 5, 6]:
        nx_coarse = (nx - 1) // factor
        ny_coarse = (ny - 1) // factor  
        nz_coarse = (nz - 1) // factor
        
        if nx_coarse >= 2 and ny_coarse >= 2 and nz_coarse >= 2:
            new_cells = (nx_coarse-1) * (ny_coarse-1) * (nz_coarse-1)
            reduction = original_cells / new_cells
            
            print(f"Facteur {factor}: {nx_coarse-1}×{ny_coarse-1}×{nz_coarse-1} = {new_cells:,} cellules (réduction {reduction:.1f}x)")
        else:
            print(f"Facteur {factor}: Trop agressif - dimensions < 2")

if __name__ == "__main__":
    test_different_coarse_factors()
    
    print("\n" + "="*70)
    input("Appuyez sur Entrée pour créer le maillage grossier...")
    
    result = create_coarse_msh_from_vtk()
    
    if result:
        print(f"\nPROCHAINES ÉTAPES :")
        print(f"1. Vérifier le maillage dans gmsh : gmsh {result['msh_file']}")
        print(f"2. lancer la simulation : python -m applications.polycrystal_ExaCA.main_traction_inconel625_2")

