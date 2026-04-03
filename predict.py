"""
预测脚本：加载训练好的 DeepONet，对指定工况和时刻进行 PSD 预测。

用法示例:
    python predict.py                          # 使用默认参数
    python predict.py --sheet CR_2_50 --times 1800 3600 5400 7200
    python predict.py --sheet CR_1_00 --times 900 2700 5400 8100 10800

支持的功能:
    1. 指定任意工况（sheet 名称）进行预测
    2. 指定任意时刻（不必是训练快照时刻）
    3. 自动与仿真真值对比（若该时刻有快照数据）
    4. 输出完整 PSD 曲线图 + 误差统计
    5. 导出预测结果为 CSV 文件
"""

import os
import argparse

from deeponet_pbe.gpu_config import setup_gpu
setup_gpu()

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deeponet_pbe.model import DeepONet


# ======================================================================
# 工具函数
# ======================================================================

def load_model_and_params(results_dir, weights_path=None):
    """加载模型权重和归一化参数。

    Parameters
    ----------
    results_dir : str
        训练结果目录（含 norm_params.npz）。
    weights_path : str | None
        自定义权重路径。None 则使用 results_dir/model/deeponet。
        示例: results/model/ckpt_epoch_0150/deeponet
    """
    params = np.load(
        os.path.join(results_dir, "norm_params.npz"), allow_pickle=True
    )

    branch_dim = int(params["branch_dim"])
    branch_hiddens = list(params["branch_hiddens"])
    trunk_hiddens = list(params["trunk_hiddens"])
    latent_dim = int(params["latent_dim"])

    model = DeepONet(
        branch_input_dim=branch_dim,
        trunk_input_dim=2,
        branch_hiddens=branch_hiddens,
        trunk_hiddens=trunk_hiddens,
        latent_dim=latent_dim,
    )
    dummy_b = tf.zeros((1, branch_dim))
    dummy_t = tf.zeros((1, 2))
    _ = model([dummy_b, dummy_t])

    ckpt_path = weights_path or os.path.join(results_dir, "model", "deeponet")
    model.load_weights(ckpt_path)
    print(f"Model loaded from {ckpt_path}")

    return model, params


def load_experiment(excel_path, sheet_name):
    """从 Excel 加载单个工况的原始数据。"""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    record = {}
    record["Time_s"] = df["Time_s"].values
    record["Temp_K"] = df["Temp_K"].values
    record["Conc"] = df["Conc"].values

    L_mid = df["L_mid_um"].dropna().values
    record["L_mid_um"] = L_mid
    n_L = len(L_mid)

    psd_cols = [c for c in df.columns if c.startswith("Time_") and c != "Time_s"]
    snapshot_times = []
    psd_snapshots = []
    for col in psd_cols:
        t_val = float(col.replace("Time_", "").replace("s", ""))
        snapshot_times.append(t_val)
        psd_snapshots.append(df[col].iloc[:n_L].values)

    record["snapshot_times"] = np.array(snapshot_times)
    record["psd"] = np.array(psd_snapshots)
    record["C0"] = record["Conc"][0]
    record["n_L0"] = record["psd"][0]

    return record


def build_branch_vector(record, params):
    """构建 Branch 输入向量（与训练时一致）。"""
    T_min = float(params["T_min"])
    T_max = float(params["T_max"])
    C_min = float(params["C_min"])
    C_max = float(params["C_max"])
    n_scale = float(params["n_scale"])
    snapshot_times = params["snapshot_times"]
    L_sensor_idx = params["L_sensor_idx"].astype(int)

    dt = record["Time_s"][1] - record["Time_s"][0]
    T_at_snapshots = []
    for t in snapshot_times:
        idx = min(int(t / dt), len(record["Temp_K"]) - 1)
        T_at_snapshots.append(record["Temp_K"][idx])
    T_sensors = (np.array(T_at_snapshots) - T_min) / (T_max - T_min + 1e-12)

    n_L0_sampled = record["n_L0"][L_sensor_idx]
    n_L0_norm = n_L0_sampled / n_scale

    C0_norm = (record["C0"] - C_min) / (C_max - C_min + 1e-12)

    return np.concatenate([T_sensors, n_L0_norm, [C0_norm]]).astype(np.float32)


def predict_psd(model, branch_vec, L_eval, t_query, params):
    """预测单个时刻的 PSD。

    Parameters
    ----------
    model : 训练好的 DeepONet
    branch_vec : (branch_dim,) Branch 输入
    L_eval : (n_L,) 评估粒径点 (μm)
    t_query : float 查询时刻 (s)
    params : 归一化参数

    Returns
    -------
    psd_pred : (n_L,) 预测的 n(L,t) (真实尺度)
    """
    L_max = float(params["L_max"])
    t_max = float(params["t_max"])
    n_scale = float(params["n_scale"])

    n_L = len(L_eval)
    L_norm = L_eval / L_max
    t_norm = t_query / t_max

    trunk_batch = np.stack(
        [L_norm, np.full(n_L, t_norm)], axis=-1
    ).astype(np.float32)
    branch_batch = np.tile(branch_vec, (n_L, 1))

    pred_norm = model([branch_batch, trunk_batch], training=False).numpy().flatten()
    return pred_norm * n_scale


def get_true_psd(record, L_eval_idx, t_query):
    """获取仿真真值 PSD（仅当 t_query 恰好是快照时刻时可用）。

    Returns
    -------
    true_psd : ndarray | None
    """
    snap_times = record["snapshot_times"]
    tol = 1.0
    match = np.where(np.abs(snap_times - t_query) < tol)[0]
    if len(match) > 0:
        idx = match[0]
        return record["psd"][idx][L_eval_idx]
    return None


# ======================================================================
# 可视化
# ======================================================================

def plot_single_prediction(L_eval, pred, true, t_query, sheet_name, save_path):
    """绘制单时刻 PSD 预测 vs 真值。"""
    plt.figure(figsize=(9, 5))
    if true is not None:
        plt.plot(L_eval, true, "b-", linewidth=1.5, label="Simulation (FVM)")
    plt.plot(L_eval, pred, "r--", linewidth=1.5, label="DeepONet Prediction")
    plt.xlabel("$L$ (μm)")
    plt.ylabel("$n(L,t)$")
    plt.title(f"PSD Prediction at t = {t_query:.0f}s  [{sheet_name}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_multi_time_prediction(L_eval, predictions, times, sheet_name, save_path):
    """绘制多时刻 PSD 预测叠加图。"""
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis
    n = len(times)
    for i, (t, pred) in enumerate(zip(times, predictions)):
        color = cmap(i / max(n - 1, 1))
        plt.plot(L_eval, pred, color=color, linewidth=1.2,
                 label=f"t={t:.0f}s")
    plt.xlabel("$L$ (μm)")
    plt.ylabel("$n(L,t)$")
    plt.title(f"Predicted PSD Evolution [{sheet_name}]")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Evolution plot saved: {save_path}")


# ======================================================================
# 主流程
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepONet PBE Prediction")
    parser.add_argument(
        "--sheet", type=str, default="CR_2_50",
        help="工况名称（Excel sheet 名），默认 CR_2_50",
    )
    parser.add_argument(
        "--times", type=float, nargs="+",
        default=[900, 1800, 3600, 5400, 7200, 9000, 10800],
        help="预测时刻列表 (秒)，默认 [900,1800,3600,5400,7200,9000,10800]",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="训练结果目录 (含 model/ 和 norm_params.npz)，默认 results/",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="预测输出目录，默认 predictions/",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="自定义权重路径，如 results/model/ckpt_epoch_0150/deeponet",
    )
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = args.results_dir or os.path.join(SCRIPT_DIR, "results")
    OUTPUT_DIR = args.output_dir or os.path.join(SCRIPT_DIR, "predictions")
    EXCEL_PATH = os.path.join(SCRIPT_DIR, "..", "learn", "Simulation_Results_Parallel.xlsx")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 加载模型
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading trained model...")
    model, params = load_model_and_params(RESULTS_DIR, weights_path=args.weights)

    L_eval = params["L_eval"]
    L_eval_idx = params["L_eval_idx"].astype(int)
    n_L = len(L_eval)

    # ------------------------------------------------------------------
    # 2. 加载指定工况数据
    # ------------------------------------------------------------------
    sheet_name = args.sheet
    print(f"\nLoading experiment: {sheet_name}")
    record = load_experiment(EXCEL_PATH, sheet_name)

    T_start = record["Temp_K"][0]
    T_end = record["Temp_K"][-1]
    cooling_rate = (T_end - T_start) / record["Time_s"][-1] * 3600
    print(f"  Temperature: {T_start:.2f} → {T_end:.2f} K "
          f"(cooling rate: {cooling_rate:.2f} K/h)")
    print(f"  Initial concentration C(0): {record['C0']:.6f}")

    branch_vec = build_branch_vector(record, params)

    # ------------------------------------------------------------------
    # 3. 逐时刻预测
    # ------------------------------------------------------------------
    query_times = sorted(args.times)
    print(f"\nPredicting PSD at times: {query_times}")
    print("-" * 60)

    all_predictions = []
    csv_data = {"L_um": L_eval}

    for t_query in query_times:
        pred = predict_psd(model, branch_vec, L_eval, t_query, params)
        all_predictions.append(pred)
        csv_data[f"pred_t{int(t_query)}s"] = pred

        true = get_true_psd(record, L_eval_idx, t_query)
        if true is not None:
            csv_data[f"true_t{int(t_query)}s"] = true

        # 误差统计
        if true is not None:
            mse = np.mean((pred - true) ** 2)
            mask = true > true.max() * 0.01
            rel_err = (
                np.linalg.norm(pred[mask] - true[mask])
                / (np.linalg.norm(true[mask]) + 1e-15)
            )
            print(f"  t={t_query:>8.0f}s | MSE={mse:.4e} | "
                  f"Rel.L2(active)={rel_err:.4f} | max_pred={pred.max():.4e}")
        else:
            print(f"  t={t_query:>8.0f}s | (no ground truth) | "
                  f"max_pred={pred.max():.4e}")

        # 单时刻对比图
        plot_single_prediction(
            L_eval, pred, true, t_query, sheet_name,
            save_path=os.path.join(
                OUTPUT_DIR, f"pred_{sheet_name}_t{int(t_query)}.png"
            ),
        )

    # ------------------------------------------------------------------
    # 4. 多时刻演化总图
    # ------------------------------------------------------------------
    plot_multi_time_prediction(
        L_eval, all_predictions, query_times, sheet_name,
        save_path=os.path.join(OUTPUT_DIR, f"evolution_{sheet_name}.png"),
    )

    # ------------------------------------------------------------------
    # 5. 导出 CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, f"prediction_{sheet_name}.csv")
    df_out = pd.DataFrame(csv_data)
    df_out.to_csv(csv_path, index=False, float_format="%.8e")
    print(f"\nResults exported to {csv_path}")

    print("\n" + "=" * 60)
    print("Prediction complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
