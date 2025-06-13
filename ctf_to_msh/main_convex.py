#
# Description :
#  Script de maillage direct pour les données EBSD.
#  Génère une géométrie de grains jointive via une méthode d'expansion
#  et de tracé de contours.
#
# Dépendances :
#  pip install numpy pandas "orix==0.13.3" scikit-image matplotlib shapely scipy
#
# Prérequis :
#  - Gmsh doit être installé et présent dans le PATH.
#  - Python 3.8+

import pathlib
import re
import subprocess
import os
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.measure import find_contours

from scipy.ndimage import label as nd_label, binary_fill_holes, distance_transform_edt
from scipy.spatial import ConvexHull
from scipy.stats import median_abs_deviation

from orix.quaternion import Orientation, get_point_group
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

# ===========================================================================
# PARAMETRES
# ==============================================================================
SEUIL_PIXELS_PETITS_GRAINS = 30
TOLERANCE_SIMPLIFICATION = 1.0
GMSH_MESH_SIZE_FACTOR = 0.5
MIN_FRAGMENT_AREA = 10

# ===========================================================================
# FONCTIONS AUXILIAIRES
# ==============================================================================
def parser_ctf(chemin_fichier: pathlib.Path):
    en_tete, index_ligne_noms_colonnes = {}, -1
    with open(chemin_fichier, 'r', errors='ignore') as f: lignes = f.readlines()
    for i, ligne in enumerate(lignes):
        if ligne.strip().startswith('Phase') and all(k in ligne for k in ['X', 'Y', 'Euler1']):
            index_ligne_noms_colonnes = i; break
    if index_ligne_noms_colonnes == -1: raise ValueError("Ligne d'en-tête non trouvée.")
    for ligne in lignes[:index_ligne_noms_colonnes]:
        match = re.match(r'(\w+)\s+([\d.-]+)', ligne)
        if match: en_tete[match.groups()[0]] = float(match.groups()[1]) if '.' in match.groups()[1] else int(match.groups()[1])
    noms_colonnes = re.split(r'\s+', lignes[index_ligne_noms_colonnes].strip())
    donnees_df = pd.read_csv(chemin_fichier, sep=r'\s+', skiprows=index_ligne_noms_colonnes + 1, header=None, names=noms_colonnes, on_bad_lines='warn', dtype=np.float64, engine='c')
    x_cells, y_cells = int(en_tete.get('XCells', 0)), int(en_tete.get('YCells', 0))
    if x_cells == 0 or y_cells == 0: raise ValueError("Dimensions XCells/YCells non trouvées.")
    return donnees_df, (x_cells, y_cells), get_point_group(225)

def segmenter_grains_manuellement(orientations, masque_a_ignorer, seuil_desorientation_deg=5.0):
    frontieres=np.zeros(orientations.shape,dtype=bool)
    desorient_x=np.rad2deg(orientations[:,:-1].angle_with(orientations[:,1:]).data)
    desorient_y=np.rad2deg(orientations[:-1,:].angle_with(orientations[1:,:]).data)
    frontieres[:,:-1]|=(desorient_x > seuil_desorientation_deg); frontieres[:,1:]|=(desorient_x > seuil_desorientation_deg)
    frontieres[:-1,:]|=(desorient_y > seuil_desorientation_deg); frontieres[1:,:]|=(desorient_y > seuil_desorientation_deg)
    carte_labels, n_grains = nd_label(~masque_a_ignorer & ~frontieres)
    return carte_labels, n_grains

def identifier_points_aberrants_par_mad(orientations, initial_non_indexed_mask, seuil_mad_score=3.5):
    """Calcule le KAM et identifie les points aberrants avec la méthode MAD."""
    print("[INFO] Identification des points aberrants (méthode MAD)...")
    y_cells, x_cells = orientations.shape
    kam = np.zeros(orientations.shape, dtype=float)
    euler_padded_rad = np.pad(orientations.to_euler(), ((1, 1), (1, 1), (0, 0)), mode='edge')
    orientations_padded = Orientation.from_euler(euler_padded_rad, symmetry=orientations.symmetry)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            kam += orientations.angle_with(orientations_padded[1+dy:y_cells+1+dy, 1+dx:x_cells+1+dx]).data
    kam /= 8
    kam_indexed_only = np.copy(kam)
    kam_indexed_only[initial_non_indexed_mask] = np.nan
    mad = median_abs_deviation(kam_indexed_only.flatten(), scale='normal', nan_policy='omit')
    median_kam = np.nanmedian(kam_indexed_only)
    scores_mad = np.zeros_like(kam)
    if mad > 1e-9:
        scores_mad[~np.isnan(kam_indexed_only)] = np.abs(kam_indexed_only[~np.isnan(kam_indexed_only)] - median_kam) / mad
    aberrant_mad_mask = scores_mad > seuil_mad_score
    aberrant_mad_mask[initial_non_indexed_mask] = False
    print(f"[OK] {np.sum(aberrant_mad_mask)} points aberrants identifiés et marqués pour exclusion.")
    return aberrant_mad_mask

def supprimer_grains_ilots(carte_labels: np.ndarray):
    carte_corrigee = carte_labels.copy()
    gids_a_supprimer = set()
    for gid in np.unique(carte_labels[carte_labels > 0]):
        if gid in gids_a_supprimer: continue
        coords = np.argwhere(carte_labels == gid)
        if coords.size == 0: continue
        ymin, ymax, xmin, xmax = coords[:,0].min(),coords[:,0].max(),coords[:,1].min(),coords[:,1].max()
        if ymin==0 or xmin==0 or ymax==carte_labels.shape[0]-1 or xmax==carte_labels.shape[1]-1: continue
        ids_voisins = np.setdiff1d(np.unique(carte_labels[ymin-1:ymax+2, xmin-1:xmax+2]), [0, gid])
        if len(ids_voisins) == 1:
            gids_a_supprimer.add(gid)
            carte_corrigee[carte_labels == gid] = ids_voisins[0]
    print(f"[INFO] Nettoyage des îlots : {len(gids_a_supprimer)} grains fusionnés.")
    return carte_corrigee

def calculer_orientation_moyenne_par_grain(carte_labels, orientations_brutes):
    orientations_moyennes = {}
    for grain_id in np.unique(carte_labels[carte_labels > 0]):
        orientations_du_grain = orientations_brutes[carte_labels == grain_id]
        if orientations_du_grain.size > 0:
            orientations_moyennes[grain_id] = Orientation.mean(orientations_du_grain)
    return orientations_moyennes

# ===========================================================================
# VISUALISATION ET EXPORT
# ==============================================================================
def save_grain_map_image(step_number: int, step_name: str, carte_labels: np.ndarray, orientations_brutes: Orientation, ipf_key, dossier_sortie: pathlib.Path):
    """Sauvegarde une image de la carte des grains à une étape donnée."""
    print(f"[INFO] Génération de l'image pour l'étape {step_number} : {step_name}...")
    orientations_moyennes = calculer_orientation_moyenne_par_grain(carte_labels, orientations_brutes)
    num_grains = len(orientations_moyennes)
    map_orientations = Orientation.identity(orientations_brutes.shape); map_orientations.symmetry = orientations_brutes.symmetry
    for gid, o in orientations_moyennes.items(): map_orientations[carte_labels == gid] = o
    rgb_map = ipf_key.orientation2color(map_orientations)
    rgb_map[carte_labels == 0] = [1, 1, 1]
    image_a_tracer = mark_boundaries(rgb_map, carte_labels, color=(0,0,0), mode='thick')
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_a_tracer)
    titre = f"Étape {step_number}: {step_name.replace('_', ' ').title()}\n(Nombre de Grains: {num_grains})"
    ax.set_title(titre, fontsize=16)
    ax.axis('off')
    filename = dossier_sortie / f"{step_number:02d}_{step_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Image sauvegardée : '{filename.name}'")

def export_to_geo(grains: dict, filepath: pathlib.Path, mesh_size_factor: float):
    """Exporte la géométrie des polygones dans un fichier .geo pour Gmsh."""
    print("[INFO] Création du fichier de géométrie .geo pour Gmsh...")
    point_map,point_idx,line_idx,surface_idx,line_loop_idx={},1,1,1,1
    with open(filepath,"w")as f:
        f.write(f"Mesh.CharacteristicLengthFactor={mesh_size_factor};\n\n")
        all_polys=list(grains.values())
        for poly in all_polys:
            for x,y in poly.exterior.coords:
                if(x,y)not in point_map:f.write(f"Point({point_idx})={{{x},{y},0}};\n");point_map[(x,y)]=point_idx;point_idx+=1
            for interior in poly.interiors:
                for x,y in interior.coords:
                    if(x,y)not in point_map:f.write(f"Point({point_idx})={{{x},{y},0}};\n");point_map[(x,y)]=point_idx;point_idx+=1
        for gid,poly in grains.items():
            ext_pts=[point_map[tuple(c)] for c in poly.exterior.coords];ext_lines=[]
            for i in range(len(ext_pts)-1):f.write(f"Line({line_idx})={{{ext_pts[i]},{ext_pts[i+1]}}};\n");ext_lines.append(line_idx);line_idx+=1
            f.write(f"Line Loop({line_loop_idx})={{{','.join(map(str,ext_lines))}}};\n")
            int_loops=[line_loop_idx];line_loop_idx+=1
            for interior in poly.interiors:
                int_pts=[point_map[tuple(c)] for c in interior.coords];int_lines=[]
                for i in range(len(int_pts)-1):f.write(f"Line({line_idx})={{{int_pts[i]},{int_pts[i+1]}}};\n");int_lines.append(line_idx);line_idx+=1
                f.write(f"Line Loop({line_loop_idx})={{{','.join(map(str,int_lines))}}};\n");int_loops.append(line_loop_idx);line_loop_idx+=1
            f.write(f"Plane Surface({surface_idx})={{{','.join(map(str,int_loops))}}};\n")
            f.write(f"Physical Surface(\"grain_{gid}\")={{{surface_idx}}};\n\n");surface_idx+=1
    print(f"[OK] Fichier de géométrie '{filepath.name}' créé.")

# ===========================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    print("--- Lancement du Script de Maillage Direct avec Gmsh ---")
    dossier_sortie = pathlib.Path('data_gmsh'); dossier_sortie.mkdir(exist_ok=True)
    
    # --- ÉTAPE 1 : LECTURE ET SEGMENTATION ---
    print("\n[ÉTAPE 1/5] Lecture et Segmentation Initiale")
    donnees_brutes, (x_dim, y_dim), sym_cristal = parser_ctf(next(pathlib.Path('.').glob('*.ctf')))
    orientations_brutes = Orientation.from_euler(np.deg2rad(donnees_brutes[['Euler1','Euler2','Euler3']].values.reshape((y_dim, x_dim, 3))), symmetry=sym_cristal)
    
    # Visualisation avant le filtre MAD
    masque_sans_mad = (donnees_brutes[['Euler1','Euler2','Euler3']].values == 0).all(axis=1).reshape((y_dim, x_dim))
    carte_labels_sans_mad, n_grains_sans_mad = segmenter_grains_manuellement(orientations_brutes, masque_sans_mad)
    ipf_key = IPFColorKeyTSL(sym_cristal, direction=Vector3d.zvector())
    
    # Remplissage des vides pour la visualisation initiale
    print("[INFO] Remplissage des vides pour la visualisation initiale (sans filtre MAD)...")
    distances_init, indices_init = distance_transform_edt(carte_labels_sans_mad == 0, return_indices=True)
    carte_labels_sans_mad_remplie = carte_labels_sans_mad[tuple(indices_init)]
    save_grain_map_image(0, "segmentation_sans_filtre_mad", carte_labels_sans_mad_remplie, orientations_brutes, ipf_key, dossier_sortie)

    # Application du filtre MAD pour la suite du traitement
    masque_ignore = masque_sans_mad | identifier_points_aberrants_par_mad(orientations_brutes, masque_sans_mad)
    carte_labels, n_grains = segmenter_grains_manuellement(orientations_brutes, masque_ignore)
    save_grain_map_image(1, "segmentation_apres_filtre_mad", carte_labels, orientations_brutes, ipf_key, dossier_sortie)
    
    # --- ÉTAPE 2 : NETTOYAGE GÉOMÉTRIQUE PRIMAIRE ---
    print(f"\n[ÉTAPE 2/5] Nettoyage Géométrique Primaire (Grains initiaux: {n_grains})")
    counts = np.bincount(carte_labels.flatten())
    carte_labels[np.isin(carte_labels, np.where((counts > 0) & (counts < SEUIL_PIXELS_PETITS_GRAINS))[0])] = 0
    save_grain_map_image(2, "apres_filtrage_taille", carte_labels, orientations_brutes, ipf_key, dossier_sortie)
    for gid in np.unique(carte_labels[carte_labels > 0]): carte_labels[binary_fill_holes(carte_labels == gid)] = gid
    save_grain_map_image(3, "apres_remplissage_trous", carte_labels, orientations_brutes, ipf_key, dossier_sortie)
    carte_labels = supprimer_grains_ilots(carte_labels)
    save_grain_map_image(4, "apres_suppression_ilots", carte_labels, orientations_brutes, ipf_key, dossier_sortie)
    
    # --- ÉTAPE 3 : REMPLISSAGE DES VIDES PAR EXPANSION ---
    print("\n[ÉTAPE 3/5] Remplissage des Vides Intergranulaires par Expansion")
    distances, indices = distance_transform_edt(carte_labels == 0, return_indices=True)
    carte_labels = carte_labels[tuple(indices)]
    print("[OK] Tous les vides ont été comblés par expansion des grains voisins.")
    save_grain_map_image(5, "apres_expansion_des_grains", carte_labels, orientations_brutes, ipf_key, dossier_sortie)

    # --- ÉTAPE 4 : CONVERSION EN POLYGONES ---
    print(f"\n[ÉTAPE 4/5] Conversion de la Carte en Polygones (Grains avant: {len(np.unique(carte_labels))-1})")
    grains_finaux = {}
    for gid in np.unique(carte_labels[carte_labels > 0]):
        mask = (carte_labels == gid)
        contours = find_contours(mask, 0.5)
        if not contours: continue
        
        exterior = contours[np.argmax([len(c) for c in contours])]
        poly = Polygon(exterior[:, [1, 0]]).simplify(TOLERANCE_SIMPLIFICATION, preserve_topology=True)
        
        if poly.is_valid and poly.area > MIN_FRAGMENT_AREA:
            grains_finaux[gid] = poly
    
    print(f"[OK] Conversion terminée. Nombre de polygones finaux : {len(grains_finaux)}")

    # --- ÉTAPE 5 : EXPORT ET MAILLAGE GMSH ---
    print(f"\n[ÉTAPE 5/5] Export Géométrique et Maillage (Grains: {len(grains_finaux)})")
    geo_path = dossier_sortie / "microstructure.geo"
    export_to_geo(grains_finaux, geo_path, GMSH_MESH_SIZE_FACTOR)
    
    msh_path = dossier_sortie / "microstructure.msh"
    vtk_path = dossier_sortie / "microstructure.vtk"
    try:
        cmd_mesh = ["gmsh", str(geo_path), "-2", "-o", str(msh_path)]
        print(f"[INFO] Exécution de la commande : {' '.join(cmd_mesh)}")
        subprocess.run(cmd_mesh, check=True, capture_output=True, text=True, timeout=900)
        print("[OK] Maillage Gmsh (.msh) généré avec succès.")
        
        cmd_vtk = ["gmsh", str(msh_path), "-o", str(vtk_path), "-save"]
        print(f"[INFO] Exécution de la commande : {' '.join(cmd_vtk)}")
        subprocess.run(cmd_vtk, check=True, capture_output=True, text=True, timeout=600)
        print("[OK] Fichier .vtk généré pour la visualisation.")

    except FileNotFoundError:
        print("\n!!! ERREUR CRITIQUE : La commande 'gmsh' n'a pas été trouvée.")
    except subprocess.CalledProcessError as e:
        print(f"\n!!! ERREUR GMSH !!!\nCMD: {' '.join(e.cmd)}\nCode: {e.returncode}\nstderr: {e.stderr}\n")
    except subprocess.TimeoutExpired as e:
        print(f"\n!!! ERREUR GMSH !!!\nLa commande a dépassé le temps limite.\nCMD: {' '.join(e.cmd)}\n")

    print("\n--- Pipeline terminé ---")
