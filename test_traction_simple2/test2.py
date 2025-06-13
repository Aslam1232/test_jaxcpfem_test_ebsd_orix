import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def compare_stress_strain_results():
    """
    Compare les résultats de contrainte-déformation de trois simulations :
    polycristalline, homogène, et une solution analytique.
    """
    try:
        base_data_dir = 'data' 
        
        poly_case = 'traction_cube_uniaxial_debug' 
        poly_dir = os.path.join(base_data_dir, 'numpy', poly_case)
        poly_file = os.path.join(poly_dir, 'global_poly.txt')

        analytical_case = 'traction_cube_analytical'
        analytical_dir = os.path.join(base_data_dir, 'numpy', analytical_case)
        analytical_file = os.path.join(analytical_dir, 'global_analytical.txt')

        homo_case = 'traction_cube_uniaxial_homogeneous'
        homo_dir = os.path.join(base_data_dir, 'numpy', homo_case)
        homo_file = os.path.join(homo_dir, 'global_homo.txt')

        files_to_check = {
            "Numérique Polycristallin": poly_file,
            "Analytique": analytical_file,
            "Numérique Homogène": homo_file
        }

        all_files_found = True
        for label, path in files_to_check.items():
            if not os.path.exists(path):
                print(f"Fichier non trouvé pour '{label}': {path}")
                all_files_found = False
        
        if not all_files_found:
            return

        data_poly = np.loadtxt(poly_file, skiprows=1)
        eps_poly = data_poly[:, 0]
        sigma_zz_poly = data_poly[:, 1]

        data_analytical = np.loadtxt(analytical_file, skiprows=1)
        eps_analytical = data_analytical[:, 0]
        sigma_zz_analytical = data_analytical[:, 1]
        
        data_homo = np.loadtxt(homo_file, skiprows=1)
        eps_homo = data_homo[:, 0]
        sigma_zz_homo = data_homo[:, 1]

        fig, ax_main = plt.subplots(figsize=(12, 8))
        
        ax_main.plot(eps_analytical, sigma_zz_analytical, 
                 color='red', linestyle=':', marker='^', 
                 label='Analytique (E=1.25e5 MPa)', 
                 markersize=8, linewidth=2, zorder=1, markevery=3) 
        
        ax_main.plot(eps_homo, sigma_zz_homo, 
                 color='magenta', linestyle='-', marker='s', 
                 label='Numérique Homogène (JAX-FEM)', 
                 markersize=7, linewidth=1.5, zorder=2, alpha=0.6, markevery=(2,3))

        ax_main.plot(eps_poly, sigma_zz_poly, 
                 color='blue', linestyle='--', marker='o', 
                 label='Numérique Polycristallin (JAX-FEM)', 
                 markersize=8, linewidth=1.5, zorder=3, alpha=0.7, markevery=1,
                 markerfacecolor='none', markeredgecolor='blue', markeredgewidth=1.5)

        ax_main.set_xlabel('Déformation $\epsilon_{zz}$', fontsize=12)
        ax_main.set_ylabel('Contrainte $\sigma_{zz}$ (MPa)', fontsize=12)
        ax_main.set_title('Comparaison des Courbes Contrainte-Déformation', fontsize=14)
        ax_main.legend(fontsize=10, loc='best')
        ax_main.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # --- Inset Plot ---
        zoom_eps_min, zoom_eps_max = 0.0055, 0.0065
        
        relevant_sigma_values_for_zoom = []
        all_eps_data = [eps_analytical, eps_homo, eps_poly]
        all_sigma_data = [sigma_zz_analytical, sigma_zz_homo, sigma_zz_poly]

        for eps_data, sigma_data in zip(all_eps_data, all_sigma_data):
            indices_in_zoom = (eps_data >= zoom_eps_min) & (eps_data <= zoom_eps_max)
            if np.any(indices_in_zoom):
                relevant_sigma_values_for_zoom.extend(sigma_data[indices_in_zoom])
        
        if relevant_sigma_values_for_zoom:
            sigma_min_in_zoom = np.min(relevant_sigma_values_for_zoom)
            sigma_max_in_zoom = np.max(relevant_sigma_values_for_zoom)
            sigma_range_in_zoom = sigma_max_in_zoom - sigma_min_in_zoom
            zoom_sigma_min = sigma_min_in_zoom - sigma_range_in_zoom * 0.1 - 1
            zoom_sigma_max = sigma_max_in_zoom + sigma_range_in_zoom * 0.1 + 1

            ax_inset = ax_main.inset_axes([0.12, 0.58, 0.35, 0.3])
            
            ax_inset.plot(eps_analytical, sigma_zz_analytical, color='red', linestyle=':', marker='^', markersize=5, linewidth=1.5)
            ax_inset.plot(eps_homo, sigma_zz_homo, color='magenta', linestyle='-', marker='s', markersize=4, linewidth=1, alpha=0.6)
            ax_inset.plot(eps_poly, sigma_zz_poly, color='blue', linestyle='--', marker='o', markersize=4, linewidth=1, alpha=0.7, markerfacecolor='none', markeredgecolor='blue')

            ax_inset.set_xlim(zoom_eps_min, zoom_eps_max)
            ax_inset.set_ylim(zoom_sigma_min, zoom_sigma_max)
            ax_inset.set_title('Zoom', fontsize=9)
            ax_inset.tick_params(axis='both', which='major', labelsize=8)
            ax_inset.grid(True, linestyle='--', linewidth=0.4)

            try:
                ax_main.indicate_inset_zoom(ax_inset, edgecolor="black", alpha=0.5)
            except AttributeError:
                print("[INFO] La fonction indicate_inset_zoom n'est pas disponible sur cette version de matplotlib.")
        else:
            print("[INFO] Aucune donnée dans la plage de zoom spécifiée, l'inset ne sera pas créé.")

        fig.tight_layout()
        
        output_plot_filename = "comparison_plot_superposed_inset.png" 
        plt.savefig(output_plot_filename)
        print(f"\nGraphique sauvegardé sous : {output_plot_filename}")
        
        plt.show()

        print("\n--- Analyse des valeurs finales ---")
        if eps_poly.size > 0:
            print(f"Polycristallin: à eps_zz ~ {eps_poly[-1]:.4f}, sigma_zz ~ {sigma_zz_poly[-1]:.2f} MPa")

        if eps_homo.size > 0:
            print(f"Homogène: à eps_zz ~ {eps_homo[-1]:.4f}, sigma_zz ~ {sigma_zz_homo[-1]:.2f} MPa")

        if eps_analytical.size > 0:
            print(f"Analytique: à eps_zz ~ {eps_analytical[-1]:.4f}, sigma_zz ~ {sigma_zz_analytical[-1]:.2f} MPa")
        
    except FileNotFoundError as e:
        print(f"Erreur de fichier : {e}")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == '__main__':
    compare_stress_strain_results()
