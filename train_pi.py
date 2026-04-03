"""
PI-DeepONet 训练脚本：带物理约束的结晶 PBE 算子学习。

相比 data-driven 版本 (train.py) 的改进：
    1. 硬约束: n(L,0)=n₀(L), C(0)=C₀ 自动满足, J(0,t)=B₀ 自动满足
    2. PDE 残差: ∂n/∂τ + G̃·∂n/∂ξ = 0 (通过自动微分)
    3. 通量本构: J = G̃·n (Dg=0 时)
    4. 浓度预测: 网络同时输出 C(t)，数据监督
    5. 质量守恒: C(t) + α·μ₃(t) = C(0) (通过 Gauss-Legendre 积分)
    6. 梯度裁剪: 防止训练尖峰
"""

import os

from deeponet_pbe.gpu_config import setup_gpu
setup_gpu()

import numpy as np
import tensorflow as tf

from deeponet_pbe.data import PBEDataset
from deeponet_pbe.pi_model import PIDeepONet
from deeponet_pbe.pi_trainer import PITrainer
from deeponet_pbe.kinetics import CrystallizationKinetics
from deeponet_pbe.utils import (
    plot_loss, plot_psd_comparison, plot_psd_evolution,
    plot_concentration_comparison,
)


def main():
    # ==================================================================
    # 路径与超参数
    # ==================================================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXCEL_PATH = os.path.join(SCRIPT_DIR, "..", "learn", "Simulation_Results_Parallel.xlsx")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_pi")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 数据参数
    N_L_SENSORS = 100
    N_L_EVAL = 200
    TEST_SHEETS = ["CR_1_00", "CR_2_00", "CR_3_00", "CR_4_00"]

    # 网络结构
    BRANCH_HIDDENS = [256, 256, 256]
    TRUNK_HIDDENS = [128, 128, 128]
    CONC_TRUNK_HIDDENS = [64, 64]
    LATENT_DIM = 128
    USE_FLUX_OUTPUT = True
    PREDICT_CONCENTRATION = True
    BC_EPSILON = 0.001

    # 训练参数
    LEARNING_RATE = 5e-4
    DECAY_STEPS = 3000
    DECAY_RATE = 0.9
    EPOCHS = 300
    BATCH_SIZE = 4096
    COLLOC_BATCH_SIZE = 2048
    CONC_BATCH_SIZE = 512
    MASS_BATCH_SIZE = 256
    PRINT_EVERY = 20
    SAVE_EVERY = 50
    GRAD_CLIP = 1.0

    # 损失权重
    LAMBDA_DATA = 1.0
    LAMBDA_PDE = 0.1
    LAMBDA_FLUX = 0.1
    LAMBDA_CONC = 1.0
    LAMBDA_MASS = 0.1

    # 配点参数
    N_COLLOC_PER_SHEET = 2000
    N_MASS_PER_SHEET = 500
    N_QUAD = 32

    # ==================================================================
    # 数据加载
    # ==================================================================
    print("=" * 60)
    print("[PI-DeepONet] Loading data from Excel...")
    dataset = PBEDataset(
        excel_path=EXCEL_PATH,
        n_L_sensors=N_L_SENSORS,
        n_L_eval=N_L_EVAL,
        test_sheets=TEST_SHEETS,
        skip_t0=True,
    )
    dataset.summary()

    print("\nBuilding training data...")
    train_data = dataset.get_train_data()
    print(f"  Branch: {train_data[0].shape}")
    print(f"  Trunk:  {train_data[1].shape}")
    print(f"  Labels: {train_data[2].shape}")

    print("Building test data...")
    test_data = dataset.get_test_data()
    print(f"  Test samples: {test_data[0].shape[0]}")

    print("\nBuilding concentration training data...")
    conc_train = dataset.get_concentration_train_data()
    print(f"  Conc train: branch={conc_train[0].shape}, "
          f"tau={conc_train[1].shape}, C={conc_train[2].shape}")

    conc_test = dataset.get_concentration_test_data()
    print(f"  Conc test:  branch={conc_test[0].shape}, "
          f"tau={conc_test[1].shape}, C={conc_test[2].shape}")

    # ==================================================================
    # 动力学模块
    # ==================================================================
    print("\nInitializing crystallization kinetics...")
    kinetics = CrystallizationKinetics(
        raw_data=dataset._raw,
        t_max=dataset.t_max,
        L_max=dataset.L_max,
        n_scale=dataset.n_scale,
        C_min=dataset.C_min,
        C_max=dataset.C_max,
    )
    print(f"  Available sheets: {len(kinetics.available_sheets)}")

    alpha_norm = kinetics.estimate_alpha_norm(
        raw_data=dataset._raw,
        snapshot_times=dataset.snapshot_times,
        L_full=dataset._L_full,
    )
    print(f"  Estimated α_norm (mass conservation coeff): {alpha_norm:.6f}")

    # ==================================================================
    # 构建模型
    # ==================================================================
    xi_sensors = dataset.L_sensors / dataset.L_max
    n_T_sensors = len(dataset.snapshot_times)

    print("=" * 60)
    print("Building PI-DeepONet model (with concentration + IC constraint)...")
    print(f"  IC hard constraint: n(xi,0) = n0(xi) from {len(xi_sensors)} PSD sensors")
    model = PIDeepONet(
        branch_input_dim=dataset.branch_dim,
        trunk_input_dim=dataset.trunk_dim,
        branch_hiddens=BRANCH_HIDDENS,
        trunk_hiddens=TRUNK_HIDDENS,
        latent_dim=LATENT_DIM,
        use_flux_output=USE_FLUX_OUTPUT,
        predict_concentration=PREDICT_CONCENTRATION,
        conc_trunk_hiddens=CONC_TRUNK_HIDDENS,
        bc_epsilon=BC_EPSILON,
        n_T_sensors=n_T_sensors,
        xi_sensors=xi_sensors,
    )
    dummy_b = tf.zeros((1, dataset.branch_dim))
    dummy_t = tf.zeros((1, dataset.trunk_dim))
    _ = model([dummy_b, dummy_t])
    model.summary()

    # ==================================================================
    # 创建训练器
    # ==================================================================
    trainer = PITrainer(
        model=model,
        kinetics=kinetics,
        learning_rate=LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        lambda_data=LAMBDA_DATA,
        lambda_pde=LAMBDA_PDE,
        lambda_flux=LAMBDA_FLUX,
        lambda_conc=LAMBDA_CONC,
        lambda_mass=LAMBDA_MASS,
        alpha_norm=alpha_norm,
        n_quad=N_QUAD,
        grad_clip_norm=GRAD_CLIP,
    )

    # ==================================================================
    # 生成配点数据
    # ==================================================================
    print("\nGenerating collocation points...")
    colloc_data = trainer.generate_collocation(
        dataset=dataset,
        n_points_per_sheet=N_COLLOC_PER_SHEET,
        rng=np.random.default_rng(42),
    )
    print(f"  PDE collocation samples: {colloc_data[0].shape[0]}")
    print(f"  G_tilde range:  [{colloc_data[2].min():.4f}, {colloc_data[2].max():.4f}]")
    print(f"  B0_norm range:  [{colloc_data[3].min():.4e}, {colloc_data[3].max():.4e}]")

    mass_colloc = trainer.generate_mass_collocation(
        dataset=dataset,
        n_points_per_sheet=N_MASS_PER_SHEET,
        rng=np.random.default_rng(43),
    )
    print(f"  Mass conservation collocation: {mass_colloc[0].shape[0]}")

    # ==================================================================
    # 训练
    # ==================================================================
    print("=" * 60)
    print("Start PI-DeepONet training (with mass conservation)...")
    print(f"  Loss weights: data={LAMBDA_DATA}, pde={LAMBDA_PDE}, "
          f"flux={LAMBDA_FLUX}, conc={LAMBDA_CONC}, mass={LAMBDA_MASS}")
    print(f"  α_norm={alpha_norm:.6f}, Gauss-Legendre Q={N_QUAD}")
    print(f"  Grad clip: {GRAD_CLIP}")
    ckpt_dir = os.path.join(OUTPUT_DIR, "model")
    trainer.fit(
        train_data=train_data,
        colloc_data=colloc_data,
        conc_train_data=conc_train,
        mass_colloc_data=mass_colloc,
        val_data=test_data,
        conc_val_data=conc_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        colloc_batch_size=COLLOC_BATCH_SIZE,
        conc_batch_size=CONC_BATCH_SIZE,
        mass_batch_size=MASS_BATCH_SIZE,
        print_every=PRINT_EVERY,
        save_dir=ckpt_dir,
        save_every=SAVE_EVERY,
    )

    # ==================================================================
    # 可视化
    # ==================================================================
    print("=" * 60)
    print("Plotting results...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 多损失曲线 ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].semilogy(trainer.history["train_data"], label="Data Loss")
    if trainer.history["val_loss"]:
        axes[0, 0].semilogy(trainer.history["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_title("PSD Data Loss"); axes[0, 0].legend()
    axes[0, 0].grid(True, which="both", ls="--", alpha=0.5)

    axes[0, 1].semilogy(trainer.history["train_pde"], label="PDE Residual", color="green")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_title("PDE Residual Loss")
    axes[0, 1].legend(); axes[0, 1].grid(True, which="both", ls="--", alpha=0.5)

    axes[0, 2].semilogy(trainer.history["train_flux"], label="Flux Residual", color="orange")
    axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_title("Flux Constitutive Loss")
    axes[0, 2].legend(); axes[0, 2].grid(True, which="both", ls="--", alpha=0.5)

    axes[1, 0].semilogy(trainer.history["train_conc"], label="Conc Data Loss", color="purple")
    if trainer.history["val_conc_loss"]:
        axes[1, 0].semilogy(trainer.history["val_conc_loss"],
                            label="Val Conc Loss", color="purple", ls="--")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("MSE")
    axes[1, 0].set_title("Concentration Data Loss"); axes[1, 0].legend()
    axes[1, 0].grid(True, which="both", ls="--", alpha=0.5)

    axes[1, 1].semilogy(trainer.history["train_mass"], label="Mass Conservation", color="red")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_title("Mass Conservation Loss")
    axes[1, 1].legend(); axes[1, 1].grid(True, which="both", ls="--", alpha=0.5)

    axes[1, 2].semilogy(trainer.history["train_total"], label="Total Loss", color="black")
    axes[1, 2].set_xlabel("Epoch"); axes[1, 2].set_title("Total Loss")
    axes[1, 2].legend(); axes[1, 2].grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pi_loss_curves.png"), dpi=150)
    plt.close()
    print(f"  Loss curves saved to {OUTPUT_DIR}/pi_loss_curves.png")

    # ---- PSD 对比图 + 浓度对比图 ----
    L_eval = dataset.L_eval
    n_L = len(L_eval)
    snapshot_times = dataset.snapshot_times
    start_idx = 1 if dataset.skip_t0 else 0
    eval_times = snapshot_times[start_idx:]
    n_times = len(eval_times)
    compare_indices = [n_times // 4, n_times // 2, 3 * n_times // 4, n_times - 1]

    for sheet_name in TEST_SHEETS:
        record = dataset._raw[sheet_name]
        branch_vec = dataset._build_branch_vector(record)

        true_all, pred_all = [], []
        C_pred_list, C_true_list, time_list = [], [], []

        for t_idx in range(n_times):
            t_val = eval_times[t_idx]
            t_norm = dataset._normalize_t(np.array([t_val]))[0]
            L_norm = dataset._normalize_L(L_eval)

            trunk_batch = np.stack(
                [L_norm, np.full(n_L, t_norm)], axis=-1
            ).astype(np.float32)
            branch_batch = np.tile(branch_vec, (n_L, 1))

            n_pred, _, C_pred = model([branch_batch, trunk_batch], B0_at_t=None)
            pred_real = dataset.inverse_normalize_n(n_pred.numpy().flatten())

            psd_idx = start_idx + t_idx
            true_real = record["psd"][psd_idx][dataset._L_eval_idx]

            true_all.append(true_real)
            pred_all.append(pred_real)

            if C_pred is not None:
                C_pred_norm = float(C_pred.numpy()[0, 0])
                C_pred_real = C_pred_norm * (dataset.C_max - dataset.C_min) + dataset.C_min
                C_pred_list.append(C_pred_real)

                dt = record["Time_s"][1] - record["Time_s"][0]
                c_idx = min(int(t_val / dt), len(record["Conc"]) - 1)
                C_true_list.append(record["Conc"][c_idx])
                time_list.append(t_val)

            if t_idx in compare_indices:
                plot_psd_comparison(
                    L_eval, true_real, pred_real,
                    time_val=t_val, sheet_name=f"{sheet_name} (PI)",
                    save_path=os.path.join(
                        OUTPUT_DIR, f"pi_psd_{sheet_name}_t{int(t_val)}.png"
                    ),
                )

        plot_psd_evolution(
            L_eval, np.array(pred_all), eval_times,
            title=f"PI-DeepONet Predicted PSD [{sheet_name}]",
            save_path=os.path.join(OUTPUT_DIR, f"pi_evolution_{sheet_name}.png"),
        )

        if C_pred_list:
            plot_concentration_comparison(
                times=np.array(time_list),
                C_true=np.array(C_true_list),
                C_pred=np.array(C_pred_list),
                sheet_name=sheet_name,
                save_path=os.path.join(
                    OUTPUT_DIR, f"pi_concentration_{sheet_name}.png"
                ),
            )

    # ---- 测试集整体误差 ----
    all_n, _, _ = model([test_data[0], test_data[1]], B0_at_t=None)
    all_pred = all_n.numpy()
    mse_norm = np.mean((all_pred - test_data[2]) ** 2)
    rel_l2 = np.linalg.norm(all_pred - test_data[2]) / (
        np.linalg.norm(test_data[2]) + 1e-12
    )
    pred_real = dataset.inverse_normalize_n(all_pred)
    true_real = dataset.inverse_normalize_n(test_data[2])
    mse_real = np.mean((pred_real - true_real) ** 2)

    print(f"\n[PSD] Test MSE (normalized):  {mse_norm:.6e}")
    print(f"[PSD] Relative L2 Error:     {rel_l2:.6e}")
    print(f"[PSD] Test MSE (real scale):  {mse_real:.6e}")

    # 浓度测试误差
    if PREDICT_CONCENTRATION and conc_test is not None:
        trunk_c_test = tf.concat(
            [tf.zeros_like(conc_test[1]), conc_test[1]], axis=-1
        )
        _, _, C_test_pred = model([conc_test[0], trunk_c_test], B0_at_t=None)
        C_test_pred = C_test_pred.numpy()
        conc_mse_norm = np.mean((C_test_pred - conc_test[2]) ** 2)
        conc_rel_l2 = np.linalg.norm(C_test_pred - conc_test[2]) / (
            np.linalg.norm(conc_test[2]) + 1e-12
        )
        print(f"\n[Conc] Test MSE (normalized): {conc_mse_norm:.6e}")
        print(f"[Conc] Relative L2 Error:    {conc_rel_l2:.6e}")

    # 保存模型
    ckpt_dir = os.path.join(OUTPUT_DIR, "model")
    model.save_weights(os.path.join(ckpt_dir, "pi_deeponet"))
    print(f"\nModel saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
