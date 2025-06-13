# test_jaxcpfem_test_ebsd_orix
# Étude de l'Anisotropie en Fabrication Additive par CPFEM

Ce dépôt contient le code développé dans le cadre d'un projet sur la modélisation de l'anisotropie mécanique des pièces issues de la fabrication additive via la méthode CPFEM.

## Contexte Technique et Attribution

Ce projet s'appuie sur le framework open-source [JAX-FEM](https://github.com/deepmodeling/jax-fem). Les simulations de plasticité cristalline sont basées sur les concepts et des éléments de code issus du solveur **[JAX-CPFEM](https://github.com/SuperkakaSCU/JAX-CPFEM)**. Les scripts présents dans ce dépôt sont des adaptations et des cas d'étude spécifiques.

## Structure du Dépôt

```
.
├── ctf_to_msh/             # Pipeline de traitement des données EBSD
└── test_traction_simple2/  # Scripts pour les simulations CPFEM (à déplacer)
```

## Installation et Prérequis

### Prérequis
- Avoir cloné ce dépôt sur votre machine locale.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda.
- Git.

### Partie 1 : Pipeline EBSD (`ctf_to_msh`)

Ce pipeline possède son propre environnement Conda.

1.  **Placez-vous dans le dossier `ctf_to_msh` du projet.**

2.  **Créez et activez l'environnement Conda :**
    ```bash
    conda env create -f environment.yml
    conda activate ebsd-pipeline-env 
    ```

### Partie 2 : Simulations CPFEM (`jax_cpfem_simulations`)

1.  **Installez l'écosystème de simulation :**
    L'exécution de ces simulations requiert une installation complète de JAX-FEM, JAX-CPFEM et Neper. La procédure d'installation est détaillée dans le rapport pdf.

2.  **Déplacez les scripts de simulation :**
    **Cette étape est indispensable.** Copiez le dossier `jax_cpfem_simulations` de ce projet dans le dossier `applications/` de votre installation locale de JAX-FEM.
    ```bash
    # Exemple de commande (à adapter selon vos chemins)
    cp -r ./jax_cpfem_simulations /chemin/vers/votre/installation/de/jax-fem/applications/
    ```

## Utilisation

### Lancer le pipeline EBSD

Une fois l'environnement `ebsd-pipeline-env` activé, placez-vous dans le dossier `ctf_to_msh` et exécutez le script principal.

```bash
# Exemple d'utilisation
python main_convex.py 
```

### Lancer une simulation CPFEM

Après avoir configuré l'environnement et déplacé les scripts :

1.  **Activez l'environnement de simulation (`jax-fem-env`).**

2.  **Placez-vous à la racine de votre installation de JAX-FEM.**
    ```bash
    cd /chemin/vers/votre/installation/de/jax-fem
    ```

3.  **Lancez la simulation** comme un module Python :
    ```bash
    # Exemple pour le test de traction uniaxiale
    python -m applications.jax_cpfem_simulations.main_ect
    ```
