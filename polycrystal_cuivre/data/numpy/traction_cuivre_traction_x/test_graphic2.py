import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Créer la figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Dimensions du rectangle (échantillon de cuivre)
width = 4.0
height = 2.0
x_start = 1.0
y_start = 1.0

# Dessiner le rectangle principal (échantillon)
rect = patches.Rectangle((x_start, y_start), width, height,
                        linewidth=3, edgecolor='black',
                        facecolor='lightcyan', alpha=0.7)
ax.add_patch(rect)

# Ajouter du texte au centre
ax.text(x_start + width/2, y_start + height/2,
        'POLYCRISTAL\nCUIVRE',
        ha='center', va='center', fontsize=14, fontweight='bold')

# Coordonnées des faces
left_face_x = x_start
right_face_x = x_start + width
bottom_y = y_start
top_y = y_start + height

# === FACE GAUCHE - BLOQUÉE EN X ===
# Dessiner des triangles pour symboliser l'encastrement
n_triangles = 5
for i in range(n_triangles):
    y_pos = bottom_y + (i + 0.5) * height / n_triangles
    triangle = patches.Polygon([
        [left_face_x - 0.15, y_pos],
        [left_face_x, y_pos - 0.1],
        [left_face_x, y_pos + 0.1]
    ], closed=True, facecolor='red', edgecolor='darkred', linewidth=2)
    ax.add_patch(triangle)

# Ligne rouge pour la face gauche
ax.plot([left_face_x, left_face_x], [bottom_y, top_y],
        'r-', linewidth=4, label='Face gauche: Ux = 0')

# === FACE DROITE - TRACTION ===
# Flèches de traction
arrow_length = 0.8
n_arrows = 3
for i in range(n_arrows):
    y_pos = bottom_y + (i + 1) * height / (n_arrows + 1)
    ax.arrow(right_face_x, y_pos, arrow_length, 0,
             head_width=0.08, head_length=0.1,
             fc='blue', ec='blue', linewidth=2)
    ax.text(right_face_x + arrow_length + 0.2, y_pos,
            f'F', ha='left', va='center',
            fontsize=12, color='blue', fontweight='bold')

# Ligne bleue pour la face droite
ax.plot([right_face_x, right_face_x], [bottom_y, top_y],
        'b-', linewidth=4, label='Face droite: Ux = δ(t)')

# === POINT DE RÉFÉRENCE - ANTI-CORPS RIGIDE ===
# Point fixe en bas à gauche
ref_point_x = left_face_x
ref_point_y = bottom_y
ax.plot(ref_point_x, ref_point_y, 'go', markersize=12, markeredgecolor='darkgreen', linewidth=3)
ax.text(ref_point_x - 0.3, ref_point_y - 0.2,
        'Point fixe:\nUx = 0, Uy = 0, Uz = 0',
        ha='center', va='top', fontsize=10,
        color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# === SYSTÈME DE COORDONNÉES ===
# Axes X, Y
origin_x, origin_y = 0.2, 0.2
axis_length = 0.6

# Axe X
ax.arrow(origin_x, origin_y, axis_length, 0,
         head_width=0.05, head_length=0.08, fc='black', ec='black')
ax.text(origin_x + axis_length + 0.1, origin_y, 'X (traction)',
        fontsize=12, va='center', fontweight='bold')

# Axe Y  
ax.arrow(origin_x, origin_y, 0, axis_length,
         head_width=0.05, head_length=0.08, fc='black', ec='black')
ax.text(origin_x, origin_y + axis_length + 0.1, 'Y',
        fontsize=12, ha='center', fontweight='bold')

# === ANNOTATIONS DÉTAILLÉES ===

# === LÉGENDE ET TITRE ===
ax.set_title('Conditions aux limites -\nSimulation de Plasticité Cristalline sur Polycristal de Cuivre',
             fontsize=16, fontweight='bold', pad=20)

# Légende personnalisée
legend_elements = [
    plt.Line2D([0], [0], color='red', lw=4, label='Face gauche bloquée (Ux = 0)'),
    plt.Line2D([0], [0], color='blue', lw=4, label='Face droite en traction (Ux = d'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Point de référence (Ux = Uy = Uz = 0)'),

]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
          bbox_to_anchor=(0.98, 0.98))


# Configuration des axes
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.8, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlabel('Position X', fontsize=12)
ax.set_ylabel('Position Y', fontsize=12)

plt.tight_layout()
plt.show()

print("🔍 RÉSUMÉ DES CONDITIONS AUX LIMITES:")
print("="*50)
print("1. FACE GAUCHE (X = X_min):")
print("   → Déplacement Ux = 0 (bloquée)")
print("   → Symbolisée par les triangles rouges")
print()
print("2. FACE DROITE (X = X_max):")
print("   → Déplacement Ux = δ(t) variable dans le temps")
print("   → Force de traction appliquée (flèches bleues)")
print()
print("3. POINT DE RÉFÉRENCE (coin bas-gauche):")
print("   → Ux = 0, Uy = 0, Uz = 0")
print("   → Évite les mouvements de corps rigide")
print()
print("4. RÉSULTAT:")
print("   → Test de traction uniaxiale pure en direction X")
print("   → Déformation cible: 0.3% (élasto-plastique)")
print("   → Observation de l'hétérogénéité entre grains")
