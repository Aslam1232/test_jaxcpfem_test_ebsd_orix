import os
import numpy as onp
import matplotlib.pyplot as plt

def problem_analytical():
    print("[ANALYTICAL] Starting analytical elastic traction calculation (Finite Strain Theory)")

    # --- Paramètres ---
    Lz = 0.02
    E = 1.25e5
    nu = 0.36
    print(f"[ANALYTICAL] Material properties: E = {E:.2e} MPa, nu = {nu:.2f}")

    case_name_numerical_to_compare = 'traction_cube_uniaxial_homogeneous'
    case_name_analytical_output = 'traction_cube_analytical'

    base_dir = os.path.dirname(__file__)
    if base_dir == "":
        base_dir = "."
    data_dir = os.path.join(base_dir, 'data')
    
    numpy_dir_numerical_to_compare = os.path.join(data_dir, 'numpy', case_name_numerical_to_compare)
    numpy_dir_analytical_output = os.path.join(data_dir, 'numpy', case_name_analytical_output)
    os.makedirs(numpy_dir_analytical_output, exist_ok=True)
    print(f"[ANALYTICAL] Ensured directory for analytical results: {numpy_dir_analytical_output}")

    num_steps = 10 
    max_eng_strain = 0.01 
    eps_values_analytical_logs = onp.linspace(0., max_eng_strain, num_steps + 1)[1:]

    global_log_analytical = []
    local_log_analytical = []
    dist_log_analytical = []
    
    print(f"[ANALYTICAL] Calculating for {len(eps_values_analytical_logs)} strain steps up to max engineering strain {eps_values_analytical_logs[-1]:.4f}")
    for eps_eng_z in eps_values_analytical_logs:
        E_zz_val = eps_eng_z + 0.5 * eps_eng_z**2
        S_zz_val = E * E_zz_val
        F_xx_sq_val = 1. - 2. * nu * E_zz_val
        F_zz_val = 1. + eps_eng_z
        if abs(F_xx_sq_val) < 1e-9:
            sigma_zz_cauchy = F_zz_val * S_zz_val 
            print(f"[ANALYTICAL] Warning: F_xx_sq_val is near zero for eps_eng_z = {eps_eng_z:.4e}")
        else:
            sigma_zz_cauchy = (F_zz_val / F_xx_sq_val) * S_zz_val
        sigma_vm_cauchy = abs(sigma_zz_cauchy)
        
        global_log_analytical.append([eps_eng_z, sigma_zz_cauchy, sigma_vm_cauchy])
        local_log_analytical.append([eps_eng_z, sigma_zz_cauchy, sigma_vm_cauchy])
        dist_log_analytical.append([eps_eng_z, sigma_zz_cauchy, sigma_zz_cauchy, sigma_zz_cauchy, sigma_zz_cauchy, sigma_zz_cauchy])
    
    headers_and_files = [
        ('eps_eng sigma_cauchy_zz_analytical_avg sigma_cauchy_vm_analytical_avg', 'global_analytical.txt', global_log_analytical),
        ('eps_eng sigma_cauchy_zz_cell0_analytical sigma_cauchy_vm_cell0_analytical', 'local_analytical.txt', local_log_analytical),
        ('eps_eng min_cauchy_zz q25 median q75 max_cauchy_zz_analytical', 'dist_analytical.txt', dist_log_analytical)
    ]
    for header, filename, data in headers_and_files:
        onp.savetxt(os.path.join(numpy_dir_analytical_output, filename), onp.array(data), header=header, fmt='%.6e')
    print(f"[ANALYTICAL] Saved analytical (finite strain) logs to {numpy_dir_analytical_output}")

    # --- Visualisation et Calcul d'Erreur ---
    analytical_label = f'Analytique (Finite Strain, E={E:.2e}, $\\nu$={nu:.2f})'

    def calculate_and_print_errors(eps_num, sigma_num, sigma_ana, label_num):
        if len(sigma_ana) != len(sigma_num):
             print(f"[ANALYTICAL] Warning: Mismatch in length for error calculation. Num: {len(sigma_num)}, Ana: {len(sigma_ana)}")
        error_abs = sigma_num - sigma_ana
        error_rel_percent = onp.zeros_like(sigma_ana)
        mask_nonzero_ana = onp.abs(sigma_ana) > 1e-9
        error_rel_percent[mask_nonzero_ana] = (error_abs[mask_nonzero_ana] / sigma_ana[mask_nonzero_ana]) * 100
        print(f"\n--- Analyse d'Erreur : Numérique ({label_num}) vs. {analytical_label} ---")
        print(f"Déformation | Num ({label_num}) | Analytique | Err. Abs. | Err. Rel. (%)")
        print("-" * 75)
        for i in range(len(eps_num)):
            rel_err_display = f"{error_rel_percent[i]:.2f}" if mask_nonzero_ana[i] else "N/A (Ana ~0)"
            print(f"{eps_num[i]:.4e} | {sigma_num[i]:.4e} | {sigma_ana[i]:.4e} | {error_abs[i]:.4e} | {rel_err_display}")
        return error_abs, error_rel_percent, mask_nonzero_ana

    print(f"\n[ANALYTICAL] Processing: Numerical Average Results from Homogeneous Simulation (global_homo.txt)")
    numerical_global_file_to_compare = os.path.join(numpy_dir_numerical_to_compare, 'global_homo.txt')
    
    if not os.path.exists(numerical_global_file_to_compare):
        print(f"[ANALYTICAL] ERROR: File not found: {numerical_global_file_to_compare}")
    else:
        try:
            data_numerical_global = onp.loadtxt(numerical_global_file_to_compare, skiprows=1)
            eps_numerical_global_eng = data_numerical_global[:, 0]
            sigma_zz_numerical_global_cauchy = data_numerical_global[:, 1]

            sigma_zz_analytical_fs_for_global_plot = []
            for eps_eng_z_num in eps_numerical_global_eng:
                E_zz_val = eps_eng_z_num + 0.5 * eps_eng_z_num**2
                S_zz_val = E * E_zz_val
                F_xx_sq_val = 1. - 2. * nu * E_zz_val
                F_zz_val = 1. + eps_eng_z_num
                if abs(F_xx_sq_val) < 1e-9:
                    sigma_val = F_zz_val * S_zz_val 
                else:
                    sigma_val = (F_zz_val / F_xx_sq_val) * S_zz_val
                sigma_zz_analytical_fs_for_global_plot.append(sigma_val)
            sigma_zz_analytical_fs_for_global_plot = onp.array(sigma_zz_analytical_fs_for_global_plot)

            plt.figure(figsize=(10, 6))
            plt.plot(eps_numerical_global_eng, sigma_zz_numerical_global_cauchy, 'bo-', label='Numérique Homogène ($\sigma_{zz}$ avg)', markersize=5)
            plt.plot(eps_numerical_global_eng, sigma_zz_analytical_fs_for_global_plot, 'r--', label=analytical_label)
            plt.xlabel('Déformation d\'ingénieur $\epsilon_{eng, zz}$'); plt.ylabel('Contrainte de Cauchy $\sigma_{zz}$ (MPa)')
            plt.title('Comparaison Numérique Homogène (Moyenne Globale) vs. Analytique (Finite Strain)'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(numpy_dir_analytical_output, 'comparison_fs_plot_global_avg_homo.png'))
            print(f"[ANALYTICAL] Saved plot: comparison_fs_plot_global_avg_homo.png")

            error_abs_g, error_rel_g, mask_g = calculate_and_print_errors(eps_numerical_global_eng, sigma_zz_numerical_global_cauchy, sigma_zz_analytical_fs_for_global_plot, "Global Avg Homo")

            plt.figure(figsize=(10, 6))
            plt.plot(eps_numerical_global_eng[mask_g], error_rel_g[mask_g], 'mo-', label='Erreur Relative Num_Global_Homo vs Ana_FS (%)', markersize=5)
            plt.xlabel('Déformation d\'ingénieur $\epsilon_{eng, zz}$'); plt.ylabel('Erreur Relative (%)')
            plt.title('Erreur Relative: Numérique Homogène (Moyenne Globale) vs. Analytique (FS)'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(numpy_dir_analytical_output, 'error_fs_plot_global_avg_homo.png'))
            print(f"[ANALYTICAL] Saved plot: error_fs_plot_global_avg_homo.png")

        except Exception as e:
            print(f"[ANALYTICAL] ERROR processing {numerical_global_file_to_compare}: {e}")

    numerical_local_2_file_to_compare = os.path.join(numpy_dir_numerical_to_compare, 'local_2_homo.txt')

    if os.path.exists(numerical_global_file_to_compare) or os.path.exists(numerical_local_2_file_to_compare):
        print("\n[ANALYTICAL] Displaying plots...")
        plt.show()
    else:
        print("\n[ANALYTICAL] No numerical data from homogeneous simulation found to display plots.")


if __name__ == '__main__':
    import time
    start = time.time()
    problem_analytical()
    print(f"[ANALYTICAL] Total time: {time.time() - start:.2f}s")
