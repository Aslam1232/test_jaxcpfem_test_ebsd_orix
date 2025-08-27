"""
Générateur de graphiques contrainte-déformation
À placer dans le dossier numpy pour générer les PNG à partir des fichiers .txt
Version autonome - utilise uniquement les données locales
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("="*60)
print("GÉNÉRATEUR GRAPHIQUES CONTRAINTE-DÉFORMATION")
print("="*60)

def find_txt_files():
    """Trouve les fichiers .txt dans le dossier courant"""
    current_dir = Path(".")
   
    # Fichiers à chercher
    required_files = {
        'detailed_stress': 'detailed_stress.txt',
        'detailed_strain': 'detailed_strain.txt',
        'global': 'global_cuivre.txt',
        'slip': 'slip_cuivre.txt'
    }
   
    found_files = {}
    missing_files = []
   
    for key, filename in required_files.items():
        filepath = current_dir / filename
        if filepath.exists():
            found_files[key] = filepath
            print(f"✅ Trouvé : {filename}")
        else:
            missing_files.append(filename)
            print(f"❌ Manquant : {filename}")
   
    return found_files, missing_files

def load_data(found_files):
    """Charge les données depuis les fichiers .txt"""
    data = {}
   
    # Chargement des contraintes détaillées
    if 'detailed_stress' in found_files:
        try:
            stress_data = np.loadtxt(found_files['detailed_stress'])
            if stress_data.ndim == 1:
                stress_data = stress_data.reshape(1, -1)
           
            data['stress'] = {
                'step': stress_data[:, 0],
                'eps': stress_data[:, 1] * 100,  # Conversion en %
                'sigma_xx': stress_data[:, 2],
                'sigma_yy': stress_data[:, 3],
                'sigma_zz': stress_data[:, 4],
                'sigma_vm': stress_data[:, 5]
            }
            print(f"✅ Contraintes chargées : {len(stress_data)} points")
        except Exception as e:
            print(f"❌ Erreur lecture detailed_stress.txt : {e}")
   
    # Chargement du glissement
    if 'slip' in found_files:
        try:
            slip_data = np.loadtxt(found_files['slip'])
            if slip_data.ndim == 1:
                slip_data = slip_data.reshape(1, -1)
           
            data['slip'] = {
                'eps': slip_data[:, 0] * 100,  # Conversion en %
                'max_slip': slip_data[:, 1]
            }
            print(f"✅ Glissement chargé : {len(slip_data)} points")
        except Exception as e:
            print(f"❌ Erreur lecture slip_cuivre.txt : {e}")
   
    # Chargement des données globales
    if 'global' in found_files:
        try:
            global_data = np.loadtxt(found_files['global'])
            if global_data.ndim == 1:
                global_data = global_data.reshape(1, -1)
           
            data['global'] = {
                'eps': global_data[:, 0] * 100,  # Conversion en %
                'sigma_xx_avg': global_data[:, 1],
                'sigma_vm_avg': global_data[:, 2]
            }
            print(f"✅ Données globales chargées : {len(global_data)} points")
        except Exception as e:
            print(f"❌ Erreur lecture global_cuivre.txt : {e}")
   
    return data

def create_plots(data):
    """Génère tous les graphiques"""
   
    plt.style.use('default')
    plots_created = []
   
    # === GRAPHIQUE 1: TOUTES LES CONTRAINTES ===
    if 'stress' in data:
        print("\n[PLOT 1] Création du graphique toutes contraintes...")
       
        plt.figure(figsize=(12, 8))
       
        stress = data['stress']
        plt.plot(stress['eps'], stress['sigma_xx'], 'b-', linewidth=3, marker='o',
                markersize=6, label='σxx (traction)')
        plt.plot(stress['eps'], stress['sigma_yy'], 'r-', linewidth=2, marker='s',
                markersize=5, label='σyy')
        plt.plot(stress['eps'], stress['sigma_zz'], 'g-', linewidth=2, marker='^',
                markersize=5, label='σzz')
        plt.plot(stress['eps'], stress['sigma_vm'], 'm-', linewidth=2, marker='d',
                markersize=5, label='von Mises')
       
        plt.xlabel('Déformation (%)', fontsize=12)
        plt.ylabel('Contrainte (MPa)', fontsize=12)
        plt.title('Contraintes vs Déformation - Polycristal Cuivre', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
       
        filename1 = 'contraintes_all_vs_strain.png'
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(filename1)
        print(f"✅ Sauvegardé : {filename1}")
   
    # === GRAPHIQUE 2: GLISSEMENT PLASTIQUE ===
    if 'slip' in data:
        print("\n[PLOT 2] Création du graphique glissement plastique...")
       
        plt.figure(figsize=(10, 6))
       
        slip = data['slip']
        plt.semilogy(slip['eps'], slip['max_slip'], 'r-', linewidth=3, marker='o', markersize=6)
       
        plt.xlabel('Déformation (%)', fontsize=12)
        plt.ylabel('Glissement plastique maximum', fontsize=12)
        plt.title('Évolution du glissement plastique - Cuivre (échelle log)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
       
        filename2 = 'slip_evolution.png'
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(filename2)
        print(f"✅ Sauvegardé : {filename2}")
   
    # === GRAPHIQUE 3: RÉPONSE ÉLASTO-PLASTIQUE (σxx seulement) ===
    if 'stress' in data:
        print("\n[PLOT 3] Création du graphique réponse élasto-plastique...")
       
        plt.figure(figsize=(10, 6))
       
        stress = data['stress']
        plt.plot(stress['eps'], stress['sigma_xx'], 'b-', linewidth=4, marker='o', markersize=8)
       
        # Module de Young du cuivre pour ligne élastique théorique
        E = 110000  # MPa
        elastic_line = E * stress['eps'] / 100  # Conversion % -> déformation
        plt.plot(stress['eps'], elastic_line, 'k--', alpha=0.5, linewidth=2,
                label='Réponse élastique théorique')
       
        plt.xlabel('Déformation (%)', fontsize=12)
        plt.ylabel('Contrainte σxx (MPa)', fontsize=12)
        plt.title('Réponse Élasto-Plastique - Polycristal Cuivre', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
       
        filename3 = 'reponse_elasto_plastique.png'
        plt.savefig(filename3, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(filename3)
        print(f"✅ Sauvegardé : {filename3}")
   
    # === GRAPHIQUE 4: CONTRAINTE PRINCIPALE (σxx) SIMPLE ===
    if 'global' in data:
        print("\n[PLOT 4] Création du graphique contrainte principale...")
       
        plt.figure(figsize=(10, 6))
       
        global_data = data['global']
        plt.plot(global_data['eps'], global_data['sigma_xx_avg'], 'b-', linewidth=4,
                marker='o', markersize=8, label='σxx moyen')
       
        plt.xlabel('Déformation (%)', fontsize=12)
        plt.ylabel('Contrainte σxx (MPa)', fontsize=12)
        plt.title('Contrainte de Traction vs Déformation - Cuivre', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
       
        filename4 = 'contrainte_traction_simple.png'
        plt.savefig(filename4, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(filename4)
        print(f"✅ Sauvegardé : {filename4}")
   
    # === GRAPHIQUE 5: VON MISES vs TRACTION ===
    if 'global' in data:
        print("\n[PLOT 5] Création du graphique von Mises vs traction...")
       
        plt.figure(figsize=(10, 6))
       
        global_data = data['global']
        plt.plot(global_data['eps'], global_data['sigma_xx_avg'], 'b-', linewidth=3,
                marker='o', markersize=6, label='σxx (traction)')
        plt.plot(global_data['eps'], global_data['sigma_vm_avg'], 'm-', linewidth=3,
                marker='s', markersize=6, label='von Mises')
       
        plt.xlabel('Déformation (%)', fontsize=12)
        plt.ylabel('Contrainte (MPa)', fontsize=12)
        plt.title('Contraintes principales - Cuivre', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
       
        filename5 = 'von_mises_vs_traction.png'
        plt.savefig(filename5, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append(filename5)
        print(f"✅ Sauvegardé : {filename5}")
   
    return plots_created

def print_data_summary(data):
    """Affiche un résumé des données"""
    print("\n" + "="*60)
    print("RÉSUMÉ DES DONNÉES")
    print("="*60)
   
    if 'stress' in data:
        stress = data['stress']
        print(f"📊 CONTRAINTES :")
        print(f"  - Points de données : {len(stress['eps'])}")
        print(f"  - Déformation max : {np.max(stress['eps']):.3f}%")
        print(f"  - σxx max : {np.max(stress['sigma_xx']):.1f} MPa")
        print(f"  - von Mises max : {np.max(stress['sigma_vm']):.1f} MPa")
   
    if 'slip' in data:
        slip = data['slip']
        print(f"\n🔄 GLISSEMENT PLASTIQUE :")
        print(f"  - Points de données : {len(slip['eps'])}")
        print(f"  - Glissement max : {np.max(slip['max_slip']):.2e}")
        print(f"  - Glissement final : {slip['max_slip'][-1]:.2e}")
   
    if 'global' in data:
        global_data = data['global']
        print(f"\n🌍 DONNÉES GLOBALES :")
        print(f"  - Points de données : {len(global_data['eps'])}")
        print(f"  - σxx final : {global_data['sigma_xx_avg'][-1]:.1f} MPa")
        print(f"  - von Mises final : {global_data['sigma_vm_avg'][-1]:.1f} MPa")

def main():
    """Fonction principale"""
   
    print(f"📁 Dossier de travail : {os.getcwd()}")
   
    # 1. Recherche des fichiers
    print(f"\n[ÉTAPE 1] Recherche des fichiers .txt...")
    found_files, missing_files = find_txt_files()
   
    if not found_files:
        print(f"\n❌ ERREUR : Aucun fichier .txt trouvé dans le dossier courant")
        print(f"📋 Fichiers requis : detailed_stress.txt, slip_cuivre.txt, global_cuivre.txt")
        return
   
    if missing_files:
        print(f"\n⚠️  Fichiers manquants : {', '.join(missing_files)}")
        print(f"▶️  Création des graphiques avec les fichiers disponibles...")
   
    # 2. Chargement des données
    print(f"\n[ÉTAPE 2] Chargement des données...")
    data = load_data(found_files)
   
    if not data:
        print(f"\n❌ ERREUR : Aucune donnée chargée avec succès")
        return
   
    # 3. Résumé des données
    print_data_summary(data)
   
    # 4. Création des graphiques
    print(f"\n[ÉTAPE 3] Génération des graphiques...")
    plots_created = create_plots(data)
   
    # 5. Résumé final
    print(f"\n" + "="*60)
    print("GÉNÉRATION TERMINÉE")
    print("="*60)
   
    if plots_created:
        print(f"✅ {len(plots_created)} graphique(s) créé(s) :")
        for i, plot in enumerate(plots_created, 1):
            print(f"  {i}. {plot}")
       
        print(f"\n📂 Fichiers PNG sauvegardés dans : {os.getcwd()}")
        print(f"🖼️  Visualisez les graphiques directement depuis ce dossier")
    else:
        print(f"❌ Aucun graphique créé - Vérifiez les fichiers de données")
   
    print("="*60)

if __name__ == "__main__":
    main()
	

