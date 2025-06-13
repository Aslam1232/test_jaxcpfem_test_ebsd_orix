import pyvista as pv
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

# --- 1. Création du maillage ---
try:
    points = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ]) * 0.02
    
    hexa_connectivity = np.array([[0,1,2,3,4,5,6,7]])
    cells_input_dict = {pv.CellType.HEXAHEDRON: hexa_connectivity}
    mesh = pv.UnstructuredGrid(cells_input_dict, points)
    cube_side_length = 0.02
    
except Exception as e:
    print(f"Erreur lors de la création du maillage : {e}. Utilisation d'un cube par défaut.")
    cube_side_length = 0.02
    mesh = pv.Cube(center=(cube_side_length/2, cube_side_length/2, cube_side_length/2), 
                   x_length=cube_side_length, y_length=cube_side_length, z_length=cube_side_length)

# --- Configuration du Plotter ---
plotter = pv.Plotter(off_screen=True, image_scale=2) 
tol = 1e-5 

# --- Définition des couleurs ---
color_side_face_str = 'lightgrey'
color_bottom_face_str = 'deepskyblue'   
color_top_face_str = 'springgreen'      
opacity_side = 0.25 
opacity_main = 1.0  

color_side_face_rgba = to_rgba(color_side_face_str, alpha=opacity_side)
color_bottom_face_rgba = to_rgba(color_bottom_face_str, alpha=opacity_main)
color_top_face_rgba = to_rgba(color_top_face_str, alpha=opacity_main)

color_bottom_marker = 'mediumblue'   
color_origin_marker = 'red'          
color_top_marker = 'darkgreen'       

# --- 2. Extraction de la surface et coloration des faces ---
if mesh.n_points > 0 and mesh.n_cells > 0:
    surface_mesh = mesh.extract_surface()
    surface_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True, auto_orient_normals=True)
    face_centers = surface_mesh.cell_centers().points
    face_normals = surface_mesh.cell_normals 
    bounds = mesh.bounds 
    L0_geom = bounds[4]  
    Lz_geom = bounds[5]  
    face_group_scalars = np.zeros(surface_mesh.n_cells, dtype=int) 
    
    is_bottom_face = np.logical_and(
        np.isclose(face_centers[:, 2], L0_geom, atol=tol),
        np.isclose(face_normals[:, 2], -1.0, atol=tol)
    )
    face_group_scalars[is_bottom_face] = 1
    
    is_top_face = np.logical_and(
        np.isclose(face_centers[:, 2], Lz_geom, atol=tol),
        np.isclose(face_normals[:, 2], 1.0, atol=tol)
    )
    face_group_scalars[is_top_face] = 2
    
    surface_mesh.cell_data['face_group'] = face_group_scalars
    
    custom_cmap_rgba = ListedColormap([
        color_side_face_rgba,   
        color_bottom_face_rgba, 
        color_top_face_rgba     
    ])
    
    plotter.add_mesh(surface_mesh, scalars='face_group', cmap=custom_cmap_rgba, 
                     show_scalar_bar=False, 
                     show_edges=True, edge_color='black', line_width=1)
else:
    print("Le maillage est vide ou ne contient pas de cellules.")

# --- 3. Représentation des marqueurs des Conditions aux Limites ---
if mesh.n_points > 0:
    node_coords = mesh.points
    L0_nodes = mesh.bounds[4] 
    Lz_nodes = mesh.bounds[5]

    bottom_nodes_indices = np.where(np.isclose(node_coords[:, 2], L0_nodes, atol=tol))[0]
    if bottom_nodes_indices.size > 0:
        bottom_points_actor = pv.PointSet(node_coords[bottom_nodes_indices])
        plotter.add_mesh(bottom_points_actor, color=color_bottom_marker, point_size=8, render_points_as_spheres=True)

    origin_point_coords_arr = None
    origin_node_dist = np.linalg.norm(node_coords - np.array([0.,0.,0.]), axis=1)
    origin_node_id = np.argmin(origin_node_dist)
    if origin_node_id < mesh.n_points and np.isclose(origin_node_dist[origin_node_id], 0.0, atol=tol):
        origin_point_coords_arr = node_coords[origin_node_id:origin_node_id+1]
    elif mesh.n_points > 0 : 
        origin_point_coords_arr = node_coords[0:1]

    if origin_point_coords_arr is not None and origin_point_coords_arr.shape[0] > 0 :
        origin_point_actor = pv.PointSet(origin_point_coords_arr)
        plotter.add_mesh(origin_point_actor, color=color_origin_marker, point_size=20, render_points_as_spheres=True)

    top_nodes_indices = np.where(np.isclose(node_coords[:, 2], Lz_nodes, atol=tol))[0]
    if top_nodes_indices.size > 0:
        top_points_coords = node_coords[top_nodes_indices]
        displacement_direction = np.zeros_like(top_points_coords)
        displacement_direction[:, 2] = 1.0 
        plotter.add_arrows(top_points_coords, displacement_direction, mag=0.003, color=color_top_marker) 

# --- 4. Configuration de la caméra et de l'apparence finale ---
plotter.camera_position = 'iso' 
plotter.camera.azimuth += 180 
plotter.camera.zoom(0.7) 
plotter.enable_anti_aliasing()
plotter.add_axes() 

if isinstance(cube_side_length, (int, float)):
    dim_text = f"Dimensions cube : {cube_side_length:.2f} x {cube_side_length:.2f} x {cube_side_length:.2f}"
else:
    dim_text = "Dimensions cube : N/A"

plotter.add_text(dim_text, position='lower_left')

# --- 5. Sauvegarde de l'image ---
output_png = "domain_visualization.png" 
try:
    plotter.screenshot(output_png)
    print(f"Image sauvegardée sous : {output_png}")
except Exception as e:
    print(f"Erreur lors de la sauvegarde de l'image : {e}")

# --- Nettoyage ---
plotter.close()
