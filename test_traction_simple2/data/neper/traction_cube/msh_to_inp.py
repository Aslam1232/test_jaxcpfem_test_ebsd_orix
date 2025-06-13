# msh_to_inp.py
import meshio
import numpy as np

# === Chemins ===
msh_file = "domain.msh"
inp_file = "mesh.inp"

# === Lecture du .msh ===
mesh = meshio.read(msh_file)

points = mesh.points
cells = mesh.cells_dict.get("hexahedron")
if cells is None:
    raise ValueError("Ce maillage ne contient pas d'éléments HEX8.")

# === Nœuds à z=0 et z=0.02 (tolérance numérique)
z_tol = 1e-6
zmin = 0.0
zmax = 0.02

fixZ_nodes = []
topZ_nodes = []

for i, pt in enumerate(points):
    if np.isclose(pt[2], zmin, atol=z_tol):
        fixZ_nodes.append(i + 1)
    elif np.isclose(pt[2], zmax, atol=z_tol):
        topZ_nodes.append(i + 1)

# === Écriture du fichier .inp
with open(inp_file, "w") as f:
    # Nœuds
    f.write("*NODE\n")
    for i, p in enumerate(points, start=1):
        f.write(f"{i}, {p[0]}, {p[1]}, {p[2]}\n")

    # Éléments
    f.write("*ELEMENT, TYPE=C3D8, ELSET=Eall\n")
    for i, cell in enumerate(cells, start=1):
        nodes = ", ".join(str(n + 1) for n in cell)
        f.write(f"{i}, {nodes}\n")

    # Ensemble d'éléments
    f.write("*ELSET, ELSET=ALL\nEall\n")

    # Ensemble de tous les nœuds
    f.write("*NSET, NSET=ALLNODES, GENERATE\n")
    f.write(f"1, {len(points)}, 1\n")

    # === NSETS POUR LES BC ===
    def write_nset(name, node_list):
        f.write(f"*NSET, NSET={name}\n")
        for i in range(0, len(node_list), 8):
            line = ", ".join(str(n) for n in node_list[i:i+8])
            f.write(line + "\n")

    write_nset("NFIXZ", fixZ_nodes)
    write_nset("NTOPX", topZ_nodes)
    write_nset("NTOPY", topZ_nodes)
    write_nset("NTOPZ", topZ_nodes)

print("[OK] Fichier 'mesh.inp' généré avec les NSET nécessaires.")
