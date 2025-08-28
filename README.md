# JAX-CPFEM EBSD Pipeline

Pipeline de modélisation CPFEM pour l'analyse de microstructures EBSD et simulations de plasticité cristalline.

## Contexte Technique et Attribution

Ce projet s'appuie sur le framework open-source [JAX-FEM](https://github.com/deepmodeling/jax-fem). Les simulations de plasticité cristalline sont basées sur les concepts et des éléments de code issus du solveur **[JAX-CPFEM](https://github.com/SuperkakaSCU/JAX-CPFEM)**. Les scripts présents dans ce dépôt sont des adaptations et des cas d'étude spécifiques.


## Structure

```
├── ctf_to_msh/              # Pipeline EBSD vers maillage
├── nano_indentation_test/   # Simulation nano-indentation (non fonctionnel - problème contact)
├── polycrystal_cuivre/      # Simulation sur maillage EBSD
├── polycrystal_ExaCA/       # Simulation à partir de données ExaCA
├── test_traction_simple2/   # Test de traction simple
└── Installation_JAX_FEM_CPFEM_Neper.pdf   # Guide d'installation
```

## Installation

### Prérequis
- Miniconda/Anaconda
- Git

### 1. Pipeline EBSD
```bash
cd ctf_to_msh
conda env create -f ../ebsd_env.yml
conda activate ebsd-env
```

### 2. Écosystème JAX-FEM/CPFEM
Suivre le guide : `Installation_JAX_FEM_CPFEM_Neper.pdf`

### 3. Configuration simulations
Copier les dossiers de simulation dans `jax-fem/applications/`

## Utilisation

### Pipeline EBSD
```bash
conda activate ebsd-env
cd ctf_to_msh
python main_convex.py
```

### Simulations
```bash
conda activate jax-fem-env
cd /chemin/vers/jax-fem

# Exemple : simulation cuivre
python -m applications.polycrystal_cuivre.main_traction_cuivre
```
