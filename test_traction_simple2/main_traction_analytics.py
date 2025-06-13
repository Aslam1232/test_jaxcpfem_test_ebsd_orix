import os
import numpy as onp
import matplotlib.pyplot as plt

def problem_analytical():
    print("[ANALYTICAL] Starting analytical elastic traction calculation")

    # --- Paramètres de la simulation et du matériau ---
    Lx = 0.02
    Lz = 0.02
    E = 1.25e5

    case_name_numerical = 'traction_cube_debug'
    case_name_analytical = 'traction_cube_analytical'

    # --- Chemins pour les sorties ---
    base_dir = os.path.dirname(__file__)
    if base_dir == "":
        base_dir = "."
    data_dir = os.path.join(base_dir, 'data')
    numpy_dir_numerical = os.path.join(data_dir, 'numpy', case_name_numerical)
    numpy_dir_analytical = os.path.join(data_dir, 'numpy', case_name_analytical)
    os.makedirs(numpy_dir_analytical, exist_ok=True)
    print(f"[ANALYTICAL] Ensured directory for analytical results: {numpy_dir_analytical}")

    # --- Génération des niveaux de déformation ---
    num_steps = 10
    disps_values = onp.linspace(0., 0.01 * Lx, num_steps + 1)[1:]
    eps_values_analytical_logs = disps_values / Lz

    # --- Calculs et sauvegarde des logs analytiques ---
    global_log_analytical = []
    local_log_analytical = []
    dist_log_analytical = []
    slip_log_analytical = []
    print(f"[ANALYTICAL] Calculating for {len(eps_values_analytical_logs)} strain steps up to max strain {eps_values_analytical_logs[-1]:.4f}")
    for eps_log in eps_values_analytical_logs:
        sigma_zz_log = E * eps_log
        sigma_vm_log = abs(sigma_zz_log)
        global_log_analytical.append([eps_log, sigma_zz_log, sigma_vm_log])
        local_log_analytical.append([eps_log, sigma_zz_log, sigma_vm_log])
        dist_log_analytical.append([eps_log, sigma_zz_log, sigma_zz_log, sigma_zz_log, sigma_zz_log, sigma_zz_log])
        slip_log_analytical.append([eps_log, 0.0])
    
    headers_and_files = [
        ('eps sigma_analytical_avg sigma_analytical_vm', 'global_analytical.txt', global_log_analytical),
        ('eps sigma_zz_cell0_analytical sigma_vm_cell0_analytical', 'local_analytical.txt', local_log_analytical),
        ('eps min q25 median q75 max_analytical', 'dist_analytical.txt', dist_log_analytical),
        ('eps max_slip_analytical', 'slip_analytical.txt', slip_log_analytical)
    ]
    for header, filename, data in headers_and_files:
        onp.savetxt(os.path.join(numpy_dir_analytical, filename), onp.array(data), header=header, fmt='%.6e')
    print(f"[ANALYTICAL] Saved analytical logs to {numpy_dir_analytical}")

    # --- Visualisation et Calcul d'Erreur ---
    def calculate_and_print_errors(eps_num, sigma_num, sigma_ana, label_num):
        error_abs = sigma_num - sigma_ana
        error_rel_percent = onp.zeros_like(sigma_ana)
        mask_nonzero_ana = sigma_ana != 0
        error_rel_percent[mask_nonzero_ana] = (error_abs[mask_nonzero_ana] / sigma_ana[mask_nonzero_ana]) * 100
        
        print(f"\n--- Analyse d'Erreur : Numérique ({label_num}) vs. Analytique ---")
        print(f"Déformation | Num ({label_num}) | Analytique | Err. Abs. | Err. Rel. (%)")
        print("-" * 75)
        for i in range(len(eps_num)):
            rel_err_display = f"{error_rel_percent[i]:.2f}" if mask_nonzero_ana[i] else "N/A (Ana=0)"
            print(f"{eps_num[i]:.4e} | {sigma_num[i]:.4e} | {sigma_ana[i]:.4e} | {error_abs[i]:.4e} | {rel_err_display}")
        return error_abs, error_rel_percent, mask_nonzero_ana

    # Comparaison avec la moyenne numérique (global_poly.txt)
    print("\n[ANALYTICAL] Processing: Numerical Average Results (global_poly.txt)")
    numerical_global_file = os.path.join(numpy_dir_numerical, 'global_poly.txt')
    
    if not os.path.exists(numerical_global_file):
        print(f"[ANALYTICAL] ERROR: File not found: {numerical_global_file}")
    else:
        try:
            data_numerical_global = onp.loadtxt(numerical_global_file, skiprows=1)
            eps_numerical_global = data_numerical_global[:, 0]
            sigma_zz_numerical_global = data_numerical_global[:, 1]
            sigma_zz_analytical_for_global_plot = E * eps_numerical_global

            plt.figure(figsize=(10, 6))
            plt.plot(eps_numerical_global, sigma_zz_numerical_global, 'bo-', label='Numérique ($\sigma_{zz}$ avg)', markersize=5)
            plt.plot(eps_numerical_global, sigma_zz_analytical_for_global_plot, 'r--', label=f'Analytique ($E={E:.2e}$ MPa)')
            plt.xlabel('Déformation $\epsilon_{zz}$'); plt.ylabel('Contrainte $\sigma_{zz}$ (MPa)')
            plt.title('Comparaison Numérique (Moyenne Globale) vs. Analytique'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(numpy_dir_analytical, 'comparison_plot_global_avg.png'))
            print(f"[ANALYTICAL] Saved plot: comparison_plot_global_avg.png")

            error_abs_g, error_rel_g, mask_g = calculate_and_print_errors(eps_numerical_global, sigma_zz_numerical_global, sigma_zz_analytical_for_global_plot, "Global Avg")

            plt.figure(figsize=(10, 6))
            plt.plot(eps_numerical_global[mask_g], error_rel_g[mask_g], 'mo-', label='Erreur Relative Num_Global vs Ana (%)', markersize=5)
            plt.xlabel('Déformation $\epsilon_{zz}$'); plt.ylabel('Erreur Relative (%)')
            plt.title('Erreur Relative: Numérique (Moyenne Globale) vs. Analytique'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(numpy_dir_analytical, 'error_plot_global_avg.png'))
            print(f"[ANALYTICAL] Saved plot: error_plot_global_avg.png")

        except Exception as e:
            print(f"[ANALYTICAL] ERROR processing {numerical_global_file}: {e}")

    # Comparaison avec la cellule du haut (local_2_poly.txt)
    print("\n[ANALYTICAL] Processing: Numerical Top Cell Results (local_2_poly.txt)")
    numerical_local_2_file = os.path.join(numpy_dir_numerical, 'local_2_poly.txt')

    if not os.path.exists(numerical_local_2_file):
        print(f"[ANALYTICAL] ERROR: File not found: {numerical_local_2_file}")
    else:
        try:
            data_numerical_local_2 = onp.loadtxt(numerical_local_2_file, skiprows=1)
            eps_numerical_local_2 = data_numerical_local_2[:, 0]
            sigma_zz_numerical_local_2 = data_numerical_local_2[:, 1]
            sigma_zz_analytical_for_local_2_plot = E * eps_numerical_local_2

            plt.figure(figsize=(10, 6))
            plt.plot(eps_numerical_local_2, sigma_zz_numerical_local_2, 'go-', label='Numérique ($\sigma_{zz}$ cellule du haut)', markersize=5)
            plt.plot(eps_numerical_local_2, sigma_zz_analytical_for_local_2_plot, 'r--', label=f'Analytique ($E={E:.2e}$ MPa)')
            plt.xlabel('Déformation $\epsilon_{zz}$'); plt.ylabel('Contrainte $\sigma_{zz}$ (MPa)')
            plt.title('Comparaison Numérique (Cellule du Haut) vs. Analytique'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(numpy_dir_analytical, 'comparison_plot_local_top_cell.png'))
            print(f"[ANALYTICAL] Saved plot: comparison_plot_local_top_cell.png")

            error_abs_l2, error_rel_l2, mask_l2 = calculate_and_print_errors(eps_numerical_local_2, sigma_zz_numerical_local_2, sigma_zz_analytical_for_local_2_plot, "Top Cell")

            plt.figure(figsize=(10, 6))
            plt.plot(eps_numerical_local_2[mask_l2], error_rel_l2[mask_l2], 'co-', label='Erreur Relative Num_TopCell vs Ana (%)', markersize=5)
            plt.xlabel('Déformation $\epsilon_{zz}$'); plt.ylabel('Erreur Relative (%)')
            plt.title('Erreur Relative: Numérique (Cellule du Haut) vs. Analytique'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(numpy_dir_analytical, 'error_plot_local_top_cell.png'))
            print(f"[ANALYTICAL] Saved plot: error_plot_local_top_cell.png")
            
        except Exception as e:
            print(f"[ANALYTICAL] ERROR processing {numerical_local_2_file}: {e}")
    
    if os.path.exists(numerical_global_file) or os.path.exists(numerical_local_2_file):
        print("\n[ANALYTICAL] Displaying plots...")
        plt.show()
    else:
        print("\n[ANALYTICAL] No numerical data found to display plots.")

if __name__ == '__main__':
    import time
    start = time.time()
    problem_analytical()
    print(f"[ANALYTICAL] Total time: {time.time() - start:.2f}s")
