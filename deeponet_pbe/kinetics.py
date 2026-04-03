"""
结晶动力学模块：从仿真数据提取 G(t)、B₀(t)、C(t) 并提供插值。

在 PI-DeepONet 中用于：
    - 物理残差计算: PDE 中的 G(t)
    - 通量边界硬约束: B₀(t)
    - 浓度预测与质量守恒: C(t), α_norm
    - 边界 soft loss:  n(0,t) = B₀(t)/G(t)
"""

from typing import Dict, List, Optional

import numpy as np


class CrystallizationKinetics:
    """管理各实验工况的 G(t)、B₀(t)、C(t) 时间序列。

    从仿真 Excel 中提取动力学数据，
    并提供在任意时刻的线性插值。

    Parameters
    ----------
    raw_data : dict
        PBEDataset._raw 字典（sheet_name → record）。
    t_max : float
        时间归一化参数。
    L_max : float
        粒径归一化参数。
    n_scale : float
        PSD 归一化参数。
    C_min : float
        浓度归一化下界。
    C_max : float
        浓度归一化上界。
    """

    def __init__(
        self,
        raw_data: Dict,
        t_max: float,
        L_max: float,
        n_scale: float,
        C_min: float = 0.0,
        C_max: float = 1.0,
    ):
        self.t_max = t_max
        self.L_max = L_max
        self.n_scale = n_scale
        self.C_min = C_min
        self.C_max = C_max
        self._kinetics = {}

        for name, record in raw_data.items():
            time_s = record["Time_s"]
            G = record.get("Growth_Rate_G", None)
            B0 = record.get("Nuc_Rate_B", None)
            Conc = record.get("Conc", None)

            if G is None or B0 is None:
                continue

            entry = {
                "time": time_s,
                "G": G,
                "B0": B0,
            }
            if Conc is not None:
                entry["Conc"] = Conc
            self._kinetics[name] = entry

    @staticmethod
    def extract_from_excel(excel_path: str, sheet_names: List[str]) -> Dict:
        """从 Excel 直接提取 G 和 B₀（备用方法）。"""
        import pandas as pd
        xls = pd.ExcelFile(excel_path)
        kinetics = {}
        for name in sheet_names:
            df = pd.read_excel(xls, sheet_name=name,
                               usecols=["Time_s", "Growth_Rate_G", "Nuc_Rate_B"])
            kinetics[name] = {
                "time": df["Time_s"].values,
                "G": df["Growth_Rate_G"].values,
                "B0": df["Nuc_Rate_B"].values,
            }
        xls.close()
        return kinetics

    def G_at(self, sheet_name: str, t_physical: np.ndarray) -> np.ndarray:
        """在物理时间 t (秒) 处插值 G(t) [μm/s]。"""
        kin = self._kinetics[sheet_name]
        return np.interp(t_physical, kin["time"], kin["G"])

    def B0_at(self, sheet_name: str, t_physical: np.ndarray) -> np.ndarray:
        """在物理时间 t (秒) 处插值 B₀(t)。"""
        kin = self._kinetics[sheet_name]
        return np.interp(t_physical, kin["time"], kin["B0"])

    def C_at(self, sheet_name: str, t_physical: np.ndarray) -> np.ndarray:
        """在物理时间 t (秒) 处插值 C(t)。"""
        kin = self._kinetics[sheet_name]
        return np.interp(t_physical, kin["time"], kin["Conc"])

    def C_normalized(self, sheet_name: str, t_physical: np.ndarray) -> np.ndarray:
        """返回归一化的 C(t) = (C - C_min) / (C_max - C_min)。"""
        C = self.C_at(sheet_name, t_physical)
        return (C - self.C_min) / (self.C_max - self.C_min + 1e-12)

    def G_normalized(self, sheet_name: str, t_physical: np.ndarray) -> np.ndarray:
        """返回无量纲化的生长率 G * t_max / L_max。

        用于归一化坐标下的 PDE 残差:
            ∂n_norm/∂τ + G̃ · ∂n_norm/∂ξ = 0
        其中 G̃ = G · t_max / L_max。
        """
        G = self.G_at(sheet_name, t_physical)
        return G * self.t_max / self.L_max

    def B0_over_G_normalized(self, sheet_name: str, t_physical: np.ndarray) -> np.ndarray:
        """返回归一化的边界值 B₀(t)/(G(t)·n_scale)。

        用于边界 soft loss:
            n_norm(0,t) ≈ B₀/(G·n_scale)
        """
        G = self.G_at(sheet_name, t_physical)
        B0 = self.B0_at(sheet_name, t_physical)
        return B0 / (G * self.n_scale + 1e-30)

    def B0_flux_normalized(self, sheet_name: str, t_physical: np.ndarray) -> np.ndarray:
        """返回归一化的通量边界值 B₀(t)·(t_max / (L_max·n_scale))。

        用于通量硬约束 J_norm(0,t) = B₀_norm(t)。
        通量 J = G·n，在归一化坐标下:
            J_norm = n_norm · G̃ = n_norm · G·t_max/L_max
        边界 J(0,t) = B₀，归一化:
            J_norm(0,t) = B₀ · t_max / (L_max · n_scale)
        """
        B0 = self.B0_at(sheet_name, t_physical)
        return B0 * self.t_max / (self.L_max * self.n_scale + 1e-30)

    def estimate_alpha_norm(
        self,
        raw_data: Dict,
        snapshot_times: np.ndarray,
        L_full: np.ndarray,
    ) -> float:
        """从仿真数据估计归一化质量守恒系数 α_norm。

        质量守恒: C(t) = C(0) - ρ_c · k_v · μ₃(t)
        归一化后: ΔC_norm(τ) = α_norm · μ₃_norm(τ)

        使用加权最小二乘法拟合：
            α_norm = Σ(ΔC_norm · μ₃_norm) / Σ(μ₃_norm²)
        自然给大 μ₃ 的晚期时刻更高权重，避免早期数值不稳定。

        Parameters
        ----------
        raw_data : dict  所有 sheet 的原始数据
        snapshot_times : ndarray  PSD 快照时刻
        L_full : ndarray  完整 L 网格 (μm)

        Returns
        -------
        alpha_norm : float  估计的归一化质量守恒系数
        """
        delta_C = self.C_max - self.C_min + 1e-30
        xi = L_full / self.L_max
        d_xi = np.diff(xi, prepend=0)

        sum_dc_mu3 = 0.0
        sum_mu3_sq = 0.0
        n_valid = 0

        for name, record in raw_data.items():
            if name not in self._kinetics:
                continue
            C0 = record["C0"]
            C0_norm = (C0 - self.C_min) / delta_C
            psd = record["psd"]           # (n_snap, n_L)

            for t_idx in range(1, len(snapshot_times)):
                n_at_t = psd[t_idx]
                n_norm = n_at_t / self.n_scale
                mu3_norm = np.sum(xi ** 3 * n_norm * d_xi)

                if mu3_norm < 1e-10:
                    continue

                t_phys = snapshot_times[t_idx]
                C_t = float(np.interp(t_phys, record["Time_s"], record["Conc"]))
                C_t_norm = (C_t - self.C_min) / delta_C
                dC_norm = C0_norm - C_t_norm

                if dC_norm <= 0:
                    continue

                sum_dc_mu3 += dC_norm * mu3_norm
                sum_mu3_sq += mu3_norm ** 2
                n_valid += 1

        if sum_mu3_sq < 1e-30 or n_valid == 0:
            print("  [Warning] Could not estimate α_norm from data, using default 1.0")
            return 1.0

        alpha = sum_dc_mu3 / sum_mu3_sq
        print(f"  α_norm estimated from {n_valid} data points (weighted LS)")
        return float(alpha)

    @property
    def available_sheets(self) -> List[str]:
        return list(self._kinetics.keys())
