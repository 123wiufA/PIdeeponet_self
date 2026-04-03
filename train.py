"""
主训练脚本：使用 DeepONet 学习 PBE 结晶过程算子。

算子映射:  (n(L,0), C(0), T(t))  →  n(L,t)
"""

import os
import sys

from deeponet_pbe.gpu_config import setup_gpu
setup_gpu()

import numpy as np
import tensorflow as tf

from deeponet_pbe import DeepONet, PBEDataset, Trainer
from deeponet_pbe.utils import (
    plot_loss,
    plot_psd_comparison,
    plot_psd_evolution,
    plot_temperature_profiles,
)


def main():
    # ==================================================================
    # 路径与超参数
    # ==================================================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXCEL_PATH = os.path.join(SCRIPT_DIR, "..", "learn", "Simulation_Results_Parallel.xlsx")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    N_L_SENSORS = 100       # Branch: 初始 PSD 下采样 bin 数
    N_L_EVAL = 200          # Trunk:  L 轴评估下采样 bin 数
    TEST_SHEETS = ["CR_1_00", "CR_2_00", "CR_3_00", "CR_4_00"]

    BRANCH_HIDDENS = [256, 256, 256]
    TRUNK_HIDDENS = [128, 128, 128]
    LATENT_DIM = 128

    LEARNING_RATE = 1e-3
    DECAY_STEPS = 5000
    DECAY_RATE = 0.95
    EPOCHS = 300
    BATCH_SIZE = 4096
    PRINT_EVERY = 20
    SAVE_EVERY = 50

    # ==================================================================
    # 数据加载
    # ==================================================================
    print("=" * 60)
    print("Loading data from Excel...")
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
    print(f"  Branch: {test_data[0].shape}")
    print(f"  Trunk:  {test_data[1].shape}")
    print(f"  Labels: {test_data[2].shape}")

    # 绘制温度曲线
    plot_temperature_profiles(
        dataset, save_path=os.path.join(OUTPUT_DIR, "temperature_profiles.png")
    )

    # ==================================================================
    # 构建模型
    # ==================================================================
    print("=" * 60)
    print("Building DeepONet model...")
    model = DeepONet(
        branch_input_dim=dataset.branch_dim,
        trunk_input_dim=dataset.trunk_dim,
        branch_hiddens=BRANCH_HIDDENS,
        trunk_hiddens=TRUNK_HIDDENS,
        latent_dim=LATENT_DIM,
    )
    dummy_b = tf.zeros((1, dataset.branch_dim))
    dummy_t = tf.zeros((1, dataset.trunk_dim))
    _ = model([dummy_b, dummy_t])
    model.summary()

    # ==================================================================
    # 训练
    # ==================================================================
    print("=" * 60)
    print("Start training...")
    trainer = Trainer(
        model=model,
        learning_rate=LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
    )
    ckpt_dir = os.path.join(OUTPUT_DIR, "model")
    trainer.fit(
        train_data=train_data,
        val_data=test_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
        save_dir=ckpt_dir,
        save_every=SAVE_EVERY,
    )

    # ==================================================================
    # 可视化结果
    # ==================================================================
    print("=" * 60)
    print("Plotting results...")

    plot_loss(
        trainer.train_loss_history,
        trainer.val_loss_history,
        save_path=os.path.join(OUTPUT_DIR, "loss_curve.png"),
    )

    # 对每个测试工况，选几个时刻做 PSD 对比
    L_eval = dataset.L_eval
    n_L = len(L_eval)
    snapshot_times = dataset.snapshot_times
    start_idx = 1 if dataset.skip_t0 else 0
    eval_times = snapshot_times[start_idx:]
    n_times = len(eval_times)

    compare_time_indices = [
        n_times // 4,
        n_times // 2,
        3 * n_times // 4,
        n_times - 1,
    ]

    for sheet_name in TEST_SHEETS:
        record = dataset._raw[sheet_name]
        branch_vec = dataset._build_branch_vector(record)

        true_psd_all = []
        pred_psd_all = []

        for t_local_idx in range(n_times):
            t_val = eval_times[t_local_idx]
            t_norm = dataset._normalize_t(np.array([t_val]))[0]
            L_norm = dataset._normalize_L(L_eval)

            trunk_batch = np.stack(
                [L_norm, np.full(n_L, t_norm)], axis=-1
            ).astype(np.float32)
            branch_batch = np.tile(branch_vec, (n_L, 1))

            pred_norm = model([branch_batch, trunk_batch], training=False).numpy().flatten()
            pred_real = dataset.inverse_normalize_n(pred_norm)

            psd_idx = start_idx + t_local_idx
            true_real = record["psd"][psd_idx][dataset._L_eval_idx]

            true_psd_all.append(true_real)
            pred_psd_all.append(pred_real)

            if t_local_idx in compare_time_indices:
                plot_psd_comparison(
                    L_eval, true_real, pred_real,
                    time_val=t_val,
                    sheet_name=sheet_name,
                    save_path=os.path.join(
                        OUTPUT_DIR, f"psd_{sheet_name}_t{int(t_val)}.png"
                    ),
                )

        # PSD 演化对比图
        true_psd_all = np.array(true_psd_all)
        pred_psd_all = np.array(pred_psd_all)

        plot_psd_evolution(
            L_eval, true_psd_all, eval_times,
            title=f"True PSD Evolution [{sheet_name}]",
            save_path=os.path.join(OUTPUT_DIR, f"evolution_true_{sheet_name}.png"),
        )
        plot_psd_evolution(
            L_eval, pred_psd_all, eval_times,
            title=f"Predicted PSD Evolution [{sheet_name}]",
            save_path=os.path.join(OUTPUT_DIR, f"evolution_pred_{sheet_name}.png"),
        )

    # 测试集整体误差
    all_pred = model([test_data[0], test_data[1]], training=False).numpy()
    mse_norm = np.mean((all_pred - test_data[2]) ** 2)
    rel_l2 = np.linalg.norm(all_pred - test_data[2]) / (
        np.linalg.norm(test_data[2]) + 1e-12
    )
    print(f"\nTest MSE (normalized):  {mse_norm:.6e}")
    print(f"Relative L2 Error:     {rel_l2:.6e}")

    # 还原到真实尺度的误差
    pred_real = dataset.inverse_normalize_n(all_pred)
    true_real = dataset.inverse_normalize_n(test_data[2])
    mse_real = np.mean((pred_real - true_real) ** 2)
    print(f"Test MSE (real scale):  {mse_real:.6e}")

    # ==================================================================
    # 保存模型与归一化参数
    # ==================================================================
    ckpt_dir = os.path.join(OUTPUT_DIR, "model")
    model.save_weights(os.path.join(ckpt_dir, "deeponet"))
    print(f"\nModel weights saved to {ckpt_dir}")

    norm_params = {
        "T_min": dataset.T_min,
        "T_max": dataset.T_max,
        "C_min": dataset.C_min,
        "C_max": dataset.C_max,
        "L_max": dataset.L_max,
        "t_max": dataset.t_max,
        "n_scale": dataset.n_scale,
        "snapshot_times": dataset.snapshot_times,
        "L_sensors": dataset.L_sensors,
        "L_eval": dataset.L_eval,
        "L_sensor_idx": dataset._L_sensor_idx,
        "L_eval_idx": dataset._L_eval_idx,
        "n_L_sensors": N_L_SENSORS,
        "n_L_eval": N_L_EVAL,
        "branch_dim": dataset.branch_dim,
        "branch_hiddens": BRANCH_HIDDENS,
        "trunk_hiddens": TRUNK_HIDDENS,
        "latent_dim": LATENT_DIM,
    }
    np.savez(os.path.join(OUTPUT_DIR, "norm_params.npz"), **norm_params)
    print(f"Normalization params saved to {OUTPUT_DIR}/norm_params.npz")


if __name__ == "__main__":
    main()
