#
# Description :
#  Script de maillage de microstructure EBSD via une approche géométrique.
#  Le processus inclut la segmentation, le nettoyage et la décomposition
#  des grains avant de générer un maillage avec Neper.
#  Il peut subdiviser les grains trop grands, découper les grains allongés
#  et filtrer les graines trop proches pour fiabiliser le maillage.
#
# Dépendances :
#  pip install numpy pandas "orix==0.13.3" scikit-image matplotlib shapely scipy meshio
#
# Prérequis :
#  - Neper 4.10.2+ doit être installé et accessible depuis le PATH.
#  - Python 3.8+

import pathlib
import re
import subprocess
import os
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotlibPolygon
from skimage.segmentation import mark_boundaries

from scipy.ndimage import label as nd_label, binary_fill_holes
from scipy.stats import median_abs_deviation
from scipy.spatial import ConvexHull, KDTree, Voronoi

from orix.quaternion import Orientation, get_point_group
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

from shapely.geometry import Polygon, Point, LineString
from shapely.ops import triangulate, split as shapely_split
import meshio

# ===========================================================================
# PARAMETRES
# ==============================================================================
SEUIL_PIXELS_PETITS_GRAINS = 30
SEUIL_ASPECT_RATIO = 3.0
SEUIL_PROXIMITE_NEPER = 0.005
MAX_SPLIT_LEVEL = 8
MIN_FRAGMENT_AREA = 10

# Paramètres pour la subdivision des grands grains
SUBDIVISER_GRANDS_GRAINS = True  # Mettre à False pour désactiver cette étape
FACTEUR_SURFACE_MAX = 15.0      # Seuil pour considérer un grain comme "grand" (surface > surface_médiane * facteur)
DENSITE_SUBDIVISION = 0.05      # Densité des nouvelles graines dans les grands grains

# ===========================================================================
# FONCTIONS AUXILIAIRES
# ==============================================================================
def parser_ctf(chemin_fichier: pathlib.Path) -> tuple:
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

def identifier_points_aberrants_par_mad(orientations, initial_non_indexed_mask, seuil_mad_score=3.5):
    y_cells, x_cells = orientations.shape
    kam = np.zeros(orientations.shape, dtype=float)
    orientations_padded = Orientation.from_euler(np.pad(orientations.to_euler(), ((1,1),(1,1),(0,0)), mode='edge'), symmetry=orientations.symmetry)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy==0 and dx==0: continue
            kam += orientations.angle_with(orientations_padded[1+dy:y_cells+1+dy, 1+dx:x_cells+1+dx]).data
    kam /= 8
    kam_indexed_only = np.copy(kam); kam_indexed_only[initial_non_indexed_mask] = np.nan
    mad = median_abs_deviation(kam_indexed_only.flatten(), scale='normal', nan_policy='omit')
    median_kam = np.nanmedian(kam_indexed_only)
    scores_mad = np.zeros_like(kam)
    if mad > 1e-9: scores_mad[~np.isnan(kam_indexed_only)] = np.abs(kam_indexed_only[~np.isnan(kam_indexed_only)] - median_kam) / mad
    aberrant_mad_mask = scores_mad > seuil_mad_score
    aberrant_mad_mask[initial_non_indexed_mask] = False
    return aberrant_mad_mask

def segmenter_grains_manuellement(orientations, masque_a_ignorer, seuil_desorientation_deg=5.0):
    frontieres = np.zeros(orientations.shape, dtype=bool)
    desorient_x = np.rad2deg(orientations[:, :-1].angle_with(orientations[:, 1:]).data)
    desorient_y = np.rad2deg(orientations[:-1, :].angle_with(orientations[1:, :]).data)

    frontieres[:, :-1] |= (desorient_x > seuil_desorientation_deg)
    frontieres[:, 1:]  |= (desorient_x > seuil_desorientation_deg)
    frontieres[:-1, :] |= (desorient_y > seuil_desorientation_deg)
    frontieres[1:, :]  |= (desorient_y > seuil_desorientation_deg)
    carte_labels, n_grains = nd_label(~masque_a_ignorer & ~frontieres)
    return carte_labels, n_grains

def calculer_orientation_moyenne_par_grain(carte_labels, orientations_originales):
    orientations_moyennes = {grain_id: Orientation.mean(orientations_originales[carte_labels == grain_id])
                             for grain_id in np.unique(carte_labels[carte_labels > 0])
                             if np.any(carte_labels == grain_id)}
    return orientations_moyennes

# ===========================================================================
# FONCTIONS DE NETTOYAGE GÉOMÉTRIQUE
# ==============================================================================

def supprimer_grains_ilots(carte_labels: np.ndarray) -> np.ndarray:
    carte_corrigee = carte_labels.copy()
    gids_a_supprimer = set()
    for gid in np.unique(carte_labels[carte_labels > 0]):
        if gid in gids_a_supprimer: continue
        coords = np.argwhere(carte_labels == gid)
        if coords.size == 0: continue
        ymin, ymax, xmin, xmax = coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()
        if ymin==0 or xmin==0 or ymax==carte_labels.shape[0]-1 or xmax==carte_labels.shape[1]-1: continue
        ids_voisins = np.setdiff1d(np.unique(carte_labels[ymin-1:ymax+2, xmin-1:xmax+2]), [0, gid])
        if len(ids_voisins) == 1:
            gids_a_supprimer.add(gid)
            carte_corrigee[carte_labels == gid] = ids_voisins[0]
    print(f"[INFO] Nettoyage des îlots : {len(gids_a_supprimer)} grains fusionnés.")
    return carte_corrigee

def subdiviser_grands_grains(grains: dict, facteur_max: float, densite: float, gid_counter: int) -> tuple[dict, int]:
    print("[INFO] Recherche et subdivision des grains excessivement grands...")
    areas = {gid: poly.area for gid, poly in grains.items()}
    if not areas: return grains, gid_counter
    
    median_area = np.median(list(areas.values()))
    seuil_area = median_area * facteur_max
    grands_grains_ids = {gid for gid, area in areas.items() if area > seuil_area}
    
    if not grands_grains_ids:
        print("[OK] Aucun grain n'est considéré comme excessivement grand.")
        return grains, gid_counter
        
    print(f"[ATTENTION] {len(grands_grains_ids)} grains très grands vont être subdivisés.")
    grains_modifies = grains.copy()
    
    for gid in grands_grains_ids:
        poly = grains_modifies.pop(gid)
        minx, miny, maxx, maxy = poly.bounds
        
        pas = np.sqrt(median_area) * densite
        x_coords = np.arange(minx, maxx, pas)
        y_coords = np.arange(miny, maxy, pas)
        
        for x in x_coords:
            for y in y_coords:
                if poly.contains(Point(x, y)):
                    grains_modifies[gid_counter] = Point(x, y).buffer(1e-6)
                    gid_counter += 1
                    
    return grains_modifies, gid_counter

# ===========================================================================
# VISUALISATION ET EXPORT
# ==============================================================================
def save_grain_map_image(step_number: int, step_name: str, carte_labels: np.ndarray, orientations_brutes: Orientation, ipf_key, dossier_sortie: pathlib.Path):
    """Sauvegarde une image de la carte des grains."""
    print(f"[INFO] Génération de l'image pour l'étape {step_number} : {step_name}...")
    orientations_moyennes = calculer_orientation_moyenne_par_grain(carte_labels, orientations_brutes)
    map_orientations = Orientation.identity(orientations_brutes.shape, symmetry=orientations_brutes.symmetry)
    for gid, o in orientations_moyennes.items():
        map_orientations[carte_labels == gid] = o
    
    rgb_map = ipf_key.orientation2color(map_orientations)
    rgb_map[carte_labels == 0] = [1, 1, 1]
    
    filename = dossier_sortie / f"{step_number:02d}_{step_name}.png"
    plt.imsave(filename, mark_boundaries(rgb_map, carte_labels, color=(0,0,0), mode='thick'), dpi=300)
    print(f"[OK] Image sauvegardée : '{filename}'")

def filtrer_graines_proches(rows_for_csv: list, seuil: float):
    if not rows_for_csv: return [], set()
    df = pd.DataFrame(rows_for_csv); coords = df[['x_cent', 'y_cent']].values
    range_c = coords.max(axis=0) - coords.min(axis=0); range_c[range_c==0]=1
    paires = KDTree((coords - coords.min(axis=0))/range_c).query_pairs(r=seuil)
    if not paires: return rows_for_csv, set()
    adj={i:[] for i in range(len(df))}; [(adj[i].append(j),adj[j].append(i)) for i,j in paires]
    visites,clusters=set(),[]
    for i in range(len(df)):
        if i not in visites:
            cluster,q=[], [i]; visites.add(i)
            while q:
                u=q.pop(0); cluster.append(u)
                for v in adj[u]:
                    if v not in visites: visites.add(v); q.append(v)
            clusters.append(cluster)
    indices_a_garder = {c[0] for c in clusters}
    gids_supprimes = set(df.iloc[list(set(range(len(df))) - indices_a_garder)]['grain_id'])
    return df.iloc[list(indices_a_garder)].to_dict('records'), gids_supprimes

def creer_visualisation_voronoi(rows, path):
    if not rows: return
    df=pd.DataFrame(rows); points=df[['norm_x','norm_y']].values
    vor=Voronoi(points); regions,vertices=voronoi_finite_polygons_2d(vor)
    fig,ax=plt.subplots(figsize=(12,12))
    for r in regions: ax.add_patch(MatplotlibPolygon(vertices[r],ec='k',fc='lightblue',alpha=0.6,lw=0.8))
    ax.scatter(points[:,0],points[:,1],s=5,c='darkblue'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title('Visualisation du Diagramme de Voronoï Final'); ax.set_aspect('equal')
    plt.savefig(path,dpi=300,bbox_inches='tight'); plt.close(fig)

def voronoi_finite_polygons_2d(vor, radius=None):
    new_regions, new_vertices = [], vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None: radius = vor.points.ptp().max() * 2
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2)); all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices): new_regions.append(vertices); continue
        ridges, new_region = all_ridges.get(p1, []), [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0: v1, v2 = v2, v1
            if v1 >= 0: continue
            t = vor.points[p2] - vor.points[p1]; t /= np.linalg.norm(t); n = np.array([-t[1], t[0]])
            direction = np.sign(np.dot(vor.points[[p1, p2]].mean(axis=0) - center, n)) * n
            new_region.append(len(new_vertices)); new_vertices.append((vor.vertices[v2] + direction * radius).tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        if len(vs) == 0: continue
        c = vs.mean(axis=0)
        new_regions.append(np.asarray(new_region)[np.argsort(np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0]))].tolist())
    return new_regions, np.asarray(new_vertices)
    
# ===========================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    print("--- Lancement du Script de Traitement de Microstructure ---")
    dossier_sortie = pathlib.Path('data'); dossier_sortie.mkdir(exist_ok=True)
    
    # --- ÉTAPE 1 : LECTURE ET SEGMENTATION ---
    print("\n[ÉTAPE 1/7] Lecture et Segmentation Initiale")
    donnees_brutes, (x_dim, y_dim), sym_cristal = parser_ctf(next(pathlib.Path('.').glob('*.ctf')))
    euler_1d = donnees_brutes[['Euler1','Euler2','Euler3']].values
    orientations_brutes = Orientation.from_euler(np.deg2rad(euler_1d.reshape((y_dim, x_dim, 3))), symmetry=sym_cristal)
    masque_ignore = (euler_1d == 0).all(axis=1).reshape((y_dim, x_dim)) | identifier_points_aberrants_par_mad(orientations_brutes, (euler_1d == 0).all(axis=1).reshape((y_dim, x_dim)))
    carte_labels, n_grains = segmenter_grains_manuellement(orientations_brutes, masque_ignore)
    ipf_key = IPFColorKeyTSL(sym_cristal, direction=Vector3d.zvector())
    save_grain_map_image(1, "segmentation_initiale", carte_labels, orientations_brutes, ipf_key, dossier_sortie)

    # --- ÉTAPE 2 : NETTOYAGE GÉOMÉTRIQUE PRIMAIRE ---
    print("\n[ÉTAPE 2/7] Nettoyage Géométrique Primaire")
    counts = np.bincount(carte_labels.flatten())
    petits_grains = np.where((counts > 0) & (counts < SEUIL_PIXELS_PETITS_GRAINS))[0]
    carte_labels[np.isin(carte_labels, petits_grains)] = 0
    print(f"[INFO] Filtrage par taille : {len(petits_grains)} grains supprimés.")
    save_grain_map_image(2, "apres_filtrage_taille", carte_labels, orientations_brutes, ipf_key, dossier_sortie)
    
    for gid in np.unique(carte_labels[carte_labels > 0]): carte_labels[binary_fill_holes(carte_labels == gid)] = gid
    save_grain_map_image(3, "apres_remplissage_trous", carte_labels, orientations_brutes, ipf_key, dossier_sortie)
    
    carte_labels = supprimer_grains_ilots(carte_labels)
    save_grain_map_image(4, "apres_suppression_ilots", carte_labels, orientations_brutes, ipf_key, dossier_sortie)
    
    # --- ÉTAPE 3 : DÉCOMPOSITION GÉOMÉTRIQUE ---
    print("\n[ÉTAPE 3/7] Décomposition des Grains (Concavité et Élongation)")
    grains_a_traiter = deque()
    grains_finaux = {}
    new_gid_counter = (carte_labels.max() if carte_labels.size > 0 else 0) + 1
    for gid in np.unique(carte_labels[carte_labels>0]):
        coords = np.argwhere(carte_labels == gid)
        if len(coords) < 4: continue
        try:
            poly = Polygon(np.column_stack((coords[:,1],coords[:,0]))[ConvexHull(np.column_stack((coords[:,1],coords[:,0]))).vertices]).buffer(0)
            if poly.is_valid and not poly.is_empty:
                grains_a_traiter.append({'gid': gid, 'poly': poly, 'split_level': 0})
        except: continue
    
    processed_count = 0
    while grains_a_traiter:
        processed_count += 1
        if processed_count % 1000 == 0: print(f"[INFO] ... Formes traitées : {processed_count}, restantes : {len(grains_a_traiter)}")
        item = grains_a_traiter.popleft(); gid, poly, split_level = item['gid'], item['poly'], item['split_level']
        if split_level >= MAX_SPLIT_LEVEL: grains_finaux[gid] = poly; continue
        if not poly.equals(poly.convex_hull):
            for triangle in triangulate(poly):
                if triangle.within(poly.buffer(1e-9)):
                    grains_a_traiter.append({'gid': new_gid_counter, 'poly': triangle, 'split_level': split_level+1}); new_gid_counter+=1
            continue
        bbox = poly.minimum_rotated_rectangle
        coords_bbox = list(bbox.exterior.coords)
        edge1 = Point(coords_bbox[0]).distance(Point(coords_bbox[1])); edge2 = Point(coords_bbox[1]).distance(Point(coords_bbox[2]))
        aspect_ratio = max(edge1, edge2) / min(edge1, edge2) if min(edge1, edge2) > 1e-6 else 1.0
        if aspect_ratio > SEUIL_ASPECT_RATIO:
            p1,p2=((coords_bbox[1][0]+coords_bbox[2][0])/2,(coords_bbox[1][1]+coords_bbox[2][1])/2),((coords_bbox[3][0]+coords_bbox[0][0])/2,(coords_bbox[3][1]+coords_bbox[0][1])/2)
            if edge1<edge2: p1,p2=((coords_bbox[0][0]+coords_bbox[1][0])/2,(coords_bbox[0][1]+coords_bbox[1][1])/2),((coords_bbox[2][0]+coords_bbox[3][0])/2,(coords_bbox[2][1]+coords_bbox[3][1])/2)
            try:
                for sub_poly in list(shapely_split(poly, LineString([Point(p1), Point(p2)])).geoms):
                    if sub_poly.area > MIN_FRAGMENT_AREA:
                        grains_a_traiter.append({'gid':new_gid_counter, 'poly':sub_poly, 'split_level':split_level+1}); new_gid_counter+=1
            except: grains_finaux[gid] = poly
            continue
        grains_finaux[gid] = poly
        
    print(f"[OK] Décomposition terminée. Nombre de grains : {len(grains_finaux)}")

    # --- ÉTAPE 4 : SUBDIVISION DES GRANDS GRAINS ---
    print("\n[ÉTAPE 4/7] Subdivision des Grains Trop Grands")
    if SUBDIVISER_GRANDS_GRAINS:
        grains_finaux, new_gid_counter = subdiviser_grands_grains(grains_finaux, FACTEUR_SURFACE_MAX, DENSITE_SUBDIVISION, new_gid_counter)
        print(f"[OK] Subdivision terminée. Nombre de grains après subdivision : {len(grains_finaux)}")
        
    # --- ÉTAPE 5 : FINALISATION DES GRAINES ---
    print("\n[ÉTAPE 5/7] Finalisation et Filtrage des Graines")
    rows_for_csv = [{'grain_id': gid, 'x_cent': poly.representative_point().x, 'y_cent': poly.representative_point().y} for gid, poly in grains_finaux.items()]
    filtered_rows, gids_supprimes = filtrer_graines_proches(rows_for_csv, SEUIL_PROXIMITE_NEPER)
    print(f"[OK] Grains finaux prêts pour Neper : {len(filtered_rows)}")

    # --- ÉTAPE 6 : VISUALISATION FINALE ET EXPORT TESS ---
    print("\n[ÉTAPE 6/7] Génération des Visualisations Finales")
    carte_labels_finale = np.zeros((y_dim, x_dim), dtype=int)
    for gid, poly in grains_finaux.items():
        if gid not in gids_supprimes:
            minx,miny,maxx,maxy=map(int,poly.bounds)
            for yy in range(miny,maxy+1):
                for xx in range(minx,maxx+1):
                    if poly.contains(Point(xx,yy)): carte_labels_finale[yy, xx] = gid
    
    save_grain_map_image(5, "apres_tous_filtres", carte_labels_finale, orientations_brutes, ipf_key, dossier_sortie)
    
    x_coords,y_coords=[r['x_cent'] for r in filtered_rows],[r['y_cent'] for r in filtered_rows]
    min_x,max_x,min_y,max_y=min(x_coords),max(x_coords),min(y_coords),max(y_coords)
    width,height=max_x-min_x,max_y-min_y
    tess_path = dossier_sortie / "microstructure.tess"
    with open(tess_path, "w") as f:
        f.write("**tess version 4.0\n**seeds\n# ID      X_norm         Y_norm\n")
        for row in filtered_rows:
            row['norm_x'] = (row['x_cent'] - min_x) / width if width > 0 else 0.5
            row['norm_y'] = (row['y_cent'] - min_y) / height if height > 0 else 0.5
            f.write(f"{int(row['grain_id']):<6d}  {row['norm_x']:9.6f}  {row['norm_y']:9.6f}\n")
    print(f"[OK] Fichier de graines '{tess_path}' généré.")

    # --- ÉTAPE 7 : EXÉCUTION DE NEPER ---
    print("\n[ÉTAPE 7/7] Exécution de Neper")
    try:
        output_neper = dossier_sortie / "microstructure_neper"
        cmd_tess = ["neper", "-T", "-n", str(len(filtered_rows)), "-domain", "square(1,1)", "-loadtess", str(tess_path), "-o", str(output_neper)]
        subprocess.run(cmd_tess, check=True, capture_output=True, text=True, timeout=300)
    
        cmd_mesh = ["neper", "-M", f"{output_neper}.tess", "-elttype", "tri", "-order", "1", "-format", "msh", "-o", str(output_neper)]
        subprocess.run(cmd_mesh, check=True, capture_output=True, text=True, timeout=300)
        meshio.write(dossier_sortie/"microstructure.vtk", meshio.read(f"{output_neper}.msh"), file_format="vtk")
        print("[OK] Maillage Neper (.msh et .vtk) généré avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"\n!!! ERREUR NEPER !!!\nCMD: {' '.join(e.cmd)}\nCode: {e.returncode}\nstderr: {e.stderr}\n")
    except subprocess.TimeoutExpired as e:
        print(f"\n!!! ERREUR NEPER !!!\nLa commande a dépassé le temps limite de 5 minutes.\nCMD: {' '.join(e.cmd)}\n")

    creer_visualisation_voronoi(filtered_rows, dossier_sortie / "06_voronoi_final.png")
    
    print("\n--- Pipeline terminé ---")
