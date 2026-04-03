"""
Physics-Informed DeepONet 模型（PBE 结晶过程）。

核心功能：
    1. 双输出头: 同时输出 n(L,t) 和通量 J(L,t)
    2. 浓度预测: 独立时间 Trunk 预测 C(t)
    3. 硬约束:
       - 初值: n(ξ, 0) = n₀(ξ)   ->  n = n₀_interp(ξ) + tau * n_raw
         n₀(ξ) 通过从 branch 输入中的初始 PSD 传感器线性插值得到
       - 浓度初值: C(0) = C₀     ->  C = C₀_norm + tau * C_raw
       - 通量边界:
           J(0,t) = B0(t)        左端点成核通量
           J(Lmax,t) = 0         右端点无粒子通量
         ->  J = phi(xi)*B0(t) + xi*(1-xi)*J_raw
             phi(xi) = (1-xi)*exp(-xi/eps)   仅在 xi~0 处有值
    4. 物理残差通过自动微分计算
    5. 质量守恒: C(t) + α·μ₃(t) = C(0)  通过软约束强制

物理背景:
    1D 结晶 PBE (无生长弥散 Dg=0):
        dn/dt + G(t) * dn/dL = 0
    质量守恒:
        C(t) = C(0) - ρ_c · k_v · μ₃(t)
        μ₃(t) = ∫₀^Lmax L³ n(L,t) dL
    边界条件: J(L=0, t) = B0(t) 是成核速率
"""

from typing import List, Optional, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras


class BranchNet(keras.Model):

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(keras.layers.Dense(h, activation="tanh",
                                             input_shape=(dim,)))
            dim = h
        layers.append(keras.layers.Dense(latent_dim))
        self.net = keras.Sequential(layers)

    def call(self, x):
        return self.net(x)


class TrunkNet(keras.Model):

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(keras.layers.Dense(h, activation="tanh",
                                             input_shape=(dim,)))
            dim = h
        layers.append(keras.layers.Dense(latent_dim, activation="tanh"))
        self.net = keras.Sequential(layers)

    def call(self, y):
        return self.net(y)


class PIDeepONet(keras.Model):
    """Physics-Informed DeepONet（通量辅助 + 浓度预测版本）。

    网络原始输出 (n_raw, J_raw, C_raw) 经硬约束变换后得到 (n, J, C):
        n(ξ,τ) = n₀_interp(ξ) + tau * n_raw      # IC: n(ξ,0) = n₀(ξ)
        J(ξ,τ) = phi(xi)*B0(t) + xi*(1-xi)*J_raw  # BC: J(0,t)=B0, J(Lmax,t)=0
        C(τ)   = C0_norm + tau * C_raw              # IC: C(0) = C0
        phi(xi) = (1-xi) * exp(-xi / epsilon)

    n₀_interp(ξ) 通过线性插值从 branch 输入中的初始 PSD 传感器值获得，
    保证 τ=0 时精确满足初始粒径分布条件。

    Parameters
    ----------
    branch_input_dim : int
    trunk_input_dim  : int  (默认 2: [L_norm, t_norm])
    branch_hiddens   : list[int]
    trunk_hiddens    : list[int]
    latent_dim       : int
    use_flux_output  : bool
    predict_concentration : bool
    conc_trunk_hiddens : list[int]
    bc_epsilon : float
    n_T_sensors : int
        Branch 输入中温度传感器数量（用于定位 PSD 段起点）。
    xi_sensors : ndarray | None
        初始 PSD 传感器在 ξ = L/L_max 空间中的位置 (n_psd_sensors,)。
        提供后启用 n₀(ξ) 硬约束；不提供则退化为 n(L,0)=0。
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 2,
        branch_hiddens: Optional[List[int]] = None,
        trunk_hiddens: Optional[List[int]] = None,
        latent_dim: int = 128,
        use_flux_output: bool = True,
        predict_concentration: bool = True,
        conc_trunk_hiddens: Optional[List[int]] = None,
        bc_epsilon: float = 0.003,
        n_T_sensors: int = 0,
        xi_sensors: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if branch_hiddens is None:
            branch_hiddens = [256, 256, 256]
        if trunk_hiddens is None:
            trunk_hiddens = [128, 128, 128]
        if conc_trunk_hiddens is None:
            conc_trunk_hiddens = [64, 64]

        self.use_flux_output = use_flux_output
        self.predict_concentration = predict_concentration
        self.bc_epsilon = bc_epsilon

        # ---- 初始 PSD 硬约束参数 ----
        self._n_T_sensors = n_T_sensors
        if xi_sensors is not None:
            xi_arr = np.asarray(xi_sensors, dtype=np.float32)
            self._n_psd_sensors = len(xi_arr)
            self._xi_sensor_min = float(xi_arr[0])
            self._xi_sensor_max = float(xi_arr[-1])
            self._has_ic_data = True
        else:
            self._n_psd_sensors = 0
            self._xi_sensor_min = 0.0
            self._xi_sensor_max = 1.0
            self._has_ic_data = False

        self.branch = BranchNet(branch_input_dim, branch_hiddens, latent_dim)
        self.trunk = TrunkNet(trunk_input_dim, trunk_hiddens, latent_dim)

        self.bias_n = self.add_weight(
            name="bias_n", shape=(1,), initializer="zeros", trainable=True
        )
        self.readout_n = keras.layers.Dense(1, name="readout_n")

        if use_flux_output:
            self.readout_J = keras.layers.Dense(1, name="readout_J")
            self.bias_J = self.add_weight(
                name="bias_J", shape=(1,), initializer="zeros", trainable=True
            )

        if predict_concentration:
            self.conc_trunk = TrunkNet(
                input_dim=1,
                hidden_dims=conc_trunk_hiddens,
                latent_dim=latent_dim,
            )
            self.readout_C = keras.layers.Dense(1, name="readout_C")
            self.bias_C = self.add_weight(
                name="bias_C", shape=(1,), initializer="zeros", trainable=True
            )

    # ------------------------------------------------------------------
    # 初始 PSD 插值
    # ------------------------------------------------------------------

    def _interpolate_initial_psd(self, branch_input, xi):
        """从 branch 输入中的初始 PSD 传感器值线性插值到查询点 ξ。

        Branch 输入结构: [ T_sensors | n₀_norm_sensors | C₀_norm ]
        n₀ 传感器近似均匀分布在 [ξ_min, ξ_max] 上。

        Parameters
        ----------
        branch_input : (batch, branch_dim)
        xi : (batch, 1)  查询点 ξ ∈ [0, 1]

        Returns
        -------
        n0_interp : (batch, 1)  插值后的归一化初始 PSD
        """
        start = self._n_T_sensors
        end = start + self._n_psd_sensors
        n0_sensors = branch_input[:, start:end]         # (B, S)

        n_s = self._n_psd_sensors
        xi_min = self._xi_sensor_min
        xi_max = self._xi_sensor_max

        xi_clamped = tf.clip_by_value(xi, xi_min, xi_max)

        frac_idx = (
            (xi_clamped - xi_min) / (xi_max - xi_min + 1e-12)
            * tf.cast(n_s - 1, tf.float32)
        )
        idx_low = tf.cast(tf.floor(frac_idx), tf.int32)
        idx_low = tf.clip_by_value(idx_low, 0, n_s - 2)
        idx_high = idx_low + 1

        weight = frac_idx - tf.cast(idx_low, tf.float32)  # (B, 1)

        batch_idx = tf.expand_dims(
            tf.range(tf.shape(n0_sensors)[0]), -1
        )                                                  # (B, 1)

        lo_idx = tf.concat([batch_idx, idx_low], axis=-1)  # (B, 2)
        hi_idx = tf.concat([batch_idx, idx_high], axis=-1)

        n0_lo = tf.expand_dims(tf.gather_nd(n0_sensors, lo_idx), -1)  # (B, 1)
        n0_hi = tf.expand_dims(tf.gather_nd(n0_sensors, hi_idx), -1)

        return n0_lo + weight * (n0_hi - n0_lo)

    # ------------------------------------------------------------------
    # 原始网络输出
    # ------------------------------------------------------------------

    def _raw_outputs(self, branch_input, trunk_input):
        """计算硬约束变换前的原始网络输出。"""
        b = self.branch(branch_input)   # (batch, p)
        t = self.trunk(trunk_input)     # (batch, p)
        features = b * t                # (batch, p)

        n_raw = self.readout_n(features) + self.bias_n  # (batch, 1)

        J_raw = None
        if self.use_flux_output:
            J_raw = self.readout_J(features) + self.bias_J

        C_raw = None
        if self.predict_concentration:
            tau_input = trunk_input[:, 1:2]             # (batch, 1)
            c_t = self.conc_trunk(tau_input)             # (batch, p)
            c_features = b * c_t                         # (batch, p)
            C_raw = self.readout_C(c_features) + self.bias_C

        return n_raw, J_raw, C_raw

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def call(self, inputs, B0_at_t=None):
        """前向传播，自动施加硬约束。

        Parameters
        ----------
        inputs : (branch_input, trunk_input)
            trunk_input[:, 0] = xi  = L/L_max  in [0,1]
            trunk_input[:, 1] = tau = t/t_max  in [0,1]
        B0_at_t : (batch, 1) | None
            通量边界值 B0(t)（归一化后），用于通量硬约束。

        Returns
        -------
        n : (batch, 1)  预测的 n(L,t)（归一化值）
        J : (batch, 1) | None  预测的 J(L,t)
        C : (batch, 1) | None  预测的 C(t)（归一化值）
        """
        branch_input, trunk_input = inputs
        xi = trunk_input[:, 0:1]    # L_norm in [0, 1]
        tau = trunk_input[:, 1:2]   # t_norm in [0, 1]

        n_raw, J_raw, C_raw = self._raw_outputs(branch_input, trunk_input)

        # ---- 硬约束: 初值 n(ξ, 0) = n₀(ξ) ----
        if self._has_ic_data:
            n0_interp = self._interpolate_initial_psd(branch_input, xi)
            n = n0_interp + tau * n_raw
        else:
            n = tau * n_raw

        # ---- 硬约束: 通量边界 ----
        J = None
        if self.use_flux_output and J_raw is not None:
            if B0_at_t is not None:
                phi = (1.0 - xi) * tf.exp(-xi / self.bc_epsilon)
                J = phi * B0_at_t + xi * (1.0 - xi) * J_raw
            else:
                J = xi * (1.0 - xi) * J_raw

        # ---- 硬约束: 浓度初值 C(0) = C₀ ----
        C = None
        if self.predict_concentration and C_raw is not None:
            C0_norm = branch_input[:, -1:]      # C₀ 是 branch 输入最后一维
            C = C0_norm + tau * C_raw           # C(τ=0) = C₀_norm

        return n, J, C

    def predict_concentration_only(self, branch_input, tau):
        """仅预测浓度（不计算 n 和 J），用于高效推理。

        Parameters
        ----------
        branch_input : (batch, branch_dim)
        tau : (batch, 1)  归一化时间

        Returns
        -------
        C : (batch, 1)  预测的 C(τ)
        """
        if not self.predict_concentration:
            return None
        b = self.branch(branch_input)
        c_t = self.conc_trunk(tau)
        c_features = b * c_t
        C_raw = self.readout_C(c_features) + self.bias_C
        C0_norm = branch_input[:, -1:]
        return C0_norm + tau * C_raw
