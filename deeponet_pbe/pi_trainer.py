"""
PI-DeepONet 训练器：组合数据损失 + PDE 残差 + 通量本构残差 + 浓度数据损失 + 质量守恒残差。

PDE (归一化坐标):
    ∂n/∂τ + G̃ · ∂n/∂ξ = 0
    其中 G̃ = G · t_max / L_max

通量本构 (Dg=0):
    J = G̃ · n

边界 (通量形式):
    J(ξ=0, τ) = B̃₀(τ)        — 通过硬约束已自动满足
    J(ξ=1, τ) = 0              — 通过硬约束已自动满足

初值:
    n(ξ, τ=0) = 0              — 通过硬约束已自动满足
    C(τ=0) = C₀                — 通过硬约束已自动满足

质量守恒 (归一化坐标):
    C_norm(τ) + α_norm · μ₃_norm(τ) = C₀_norm
    μ₃_norm(τ) = ∫₀¹ ξ³ · n_norm(ξ,τ) dξ
    α_norm = ρ_c · k_v · L_max⁴ · n_scale / ΔC  (从数据估计)

因此 loss 包含:
    L_data   — PSD 数据监督 (MSE)
    L_pde    — PDE 残差 (配点法)
    L_flux   — 通量本构残差 J - G̃·n (配点法)
    L_conc   — 浓度数据监督 (MSE)
    L_mass   — 质量守恒残差 (配点法，Gauss-Legendre 积分)
"""

import os
import time
from typing import Optional, Tuple, List, Dict

import numpy as np
import tensorflow as tf

from .pi_model import PIDeepONet
from .kinetics import CrystallizationKinetics


def gauss_legendre_01(n_points: int):
    """返回 [0, 1] 上的 Gauss-Legendre 积分节点和权重。

    Parameters
    ----------
    n_points : int  积分节点数

    Returns
    -------
    nodes : (n_points,) float64
    weights : (n_points,) float64
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    return nodes.astype(np.float32), weights.astype(np.float32)


class PITrainer:
    """PI-DeepONet 训练器（含质量守恒）。

    Parameters
    ----------
    model : PIDeepONet
    kinetics : CrystallizationKinetics
    learning_rate : float
    lambda_data : float   PSD 数据损失权重
    lambda_pde : float    PDE 残差权重
    lambda_flux : float   通量本构残差权重
    lambda_conc : float   浓度数据损失权重
    lambda_mass : float   质量守恒残差权重
    alpha_norm : float    归一化质量守恒系数
    n_quad : int          Gauss-Legendre 积分节点数
    """

    def __init__(
        self,
        model: PIDeepONet,
        kinetics: CrystallizationKinetics,
        learning_rate: float = 1e-3,
        decay_steps: Optional[int] = 5000,
        decay_rate: float = 0.95,
        lambda_data: float = 1.0,
        lambda_pde: float = 0.1,
        lambda_flux: float = 0.1,
        lambda_conc: float = 1.0,
        lambda_mass: float = 0.1,
        alpha_norm: float = 1.0,
        n_quad: int = 32,
        grad_clip_norm: float = 1.0,
    ):
        self.model = model
        self.kinetics = kinetics
        self.lambda_data = lambda_data
        self.lambda_pde = lambda_pde
        self.lambda_flux = lambda_flux
        self.lambda_conc = lambda_conc
        self.lambda_mass = lambda_mass
        self.alpha_norm = alpha_norm

        xi_q, w_q = gauss_legendre_01(n_quad)
        self.xi_quad = tf.constant(xi_q.reshape(-1, 1), dtype=tf.float32)  # (Q, 1)
        self.w_quad = tf.constant(w_q.reshape(-1, 1), dtype=tf.float32)    # (Q, 1)
        self.n_quad = n_quad

        if decay_steps is not None:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=False,
            )
        else:
            lr_schedule = learning_rate

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=grad_clip_norm,
        )
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.history: Dict[str, List[float]] = {
            "train_total": [], "train_data": [],
            "train_pde": [], "train_flux": [],
            "train_conc": [], "train_mass": [],
            "val_loss": [], "val_conc_loss": [],
        }

    # ------------------------------------------------------------------
    # 物理残差计算
    # ------------------------------------------------------------------

    def _compute_physics_loss(self, branch_in, trunk_in, G_tilde, B0_norm):
        """计算 PDE 残差和通量本构残差。"""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(trunk_in)
            n, J, _ = self.model([branch_in, trunk_in], B0_at_t=B0_norm)

        dn = tape.gradient(n, trunk_in)         # (batch, 2)
        dn_dxi = dn[:, 0:1]
        dn_dtau = dn[:, 1:2]

        del tape

        pde_residual = dn_dtau + G_tilde * dn_dxi
        pde_loss = tf.reduce_mean(tf.square(pde_residual))

        if J is not None:
            flux_residual = J - G_tilde * n
            flux_loss = tf.reduce_mean(tf.square(flux_residual))
        else:
            flux_loss = tf.constant(0.0)

        return pde_loss, flux_loss

    # ------------------------------------------------------------------
    # 质量守恒损失
    # ------------------------------------------------------------------

    def _compute_mass_conservation_loss(self, branch_mass, tau_mass):
        """计算质量守恒残差。

        质量守恒 (归一化):
            C_pred(τ) + α_norm · ∫₀¹ ξ³ n_pred(ξ,τ) dξ = C₀_norm

        通过 Gauss-Legendre 积分计算 μ₃_norm。

        Parameters
        ----------
        branch_mass : (B, branch_dim)  各质量守恒配点的 branch 输入
        tau_mass    : (B, 1)           归一化时间 τ

        Returns
        -------
        mass_loss : scalar  质量守恒残差 MSE
        """
        B = tf.shape(branch_mass)[0]
        Q = self.n_quad

        # 扩展：为每个 τ 创建 Q 个积分节点 → (B*Q, 2) 的 trunk 输入
        # tau_mass: (B, 1) → (B, Q, 1)
        tau_expand = tf.tile(tf.expand_dims(tau_mass, 1), [1, Q, 1])
        # xi_quad: (Q, 1) → (1, Q, 1) → (B, Q, 1)
        xi_expand = tf.tile(tf.expand_dims(self.xi_quad, 0), [B, 1, 1])

        trunk_flat = tf.reshape(
            tf.concat([xi_expand, tau_expand], axis=-1), [B * Q, 2]
        )
        branch_flat = tf.repeat(branch_mass, Q, axis=0)  # (B*Q, branch_dim)

        n_flat, _, C_flat = self.model(
            [branch_flat, trunk_flat], B0_at_t=None
        )

        n_grid = tf.reshape(n_flat, [B, Q, 1])       # (B, Q, 1)

        # μ₃_norm = Σ w_q · ξ_q³ · n(ξ_q, τ)
        xi3 = self.xi_quad ** 3                        # (Q, 1)
        integrand = n_grid * tf.reshape(xi3, [1, Q, 1])
        w = tf.reshape(self.w_quad, [1, Q, 1])
        mu3_norm = tf.reduce_sum(integrand * w, axis=1)  # (B, 1)

        # C 预测：取每组第一个（C 不依赖 ξ，所有 Q 个值相同）
        C_pred = tf.reshape(C_flat, [B, Q, 1])[:, 0, :]  # (B, 1)

        C0_norm = branch_mass[:, -1:]  # (B, 1)

        mass_residual = C_pred + self.alpha_norm * mu3_norm - C0_norm
        mass_loss = tf.reduce_mean(tf.square(mass_residual))

        return mass_loss

    # ------------------------------------------------------------------
    # 训练步
    # ------------------------------------------------------------------

    def _train_step(
        self,
        branch_data, trunk_data, labels,
        branch_col, trunk_col, G_tilde_col, B0_norm_col,
        branch_conc, tau_conc, C_labels,
        branch_mass, tau_mass,
    ):
        trunk_col = tf.cast(trunk_col, tf.float32)

        with tf.GradientTape() as outer_tape:
            # ---- PSD 数据损失 ----
            n_pred, _, _ = self.model(
                [branch_data, trunk_data], B0_at_t=None
            )
            data_loss = self.loss_fn(labels, n_pred)

            # ---- PDE + 通量残差 ----
            pde_loss, flux_loss = self._compute_physics_loss(
                branch_col, trunk_col, G_tilde_col, B0_norm_col,
            )

            # ---- 浓度数据损失 ----
            if self.model.predict_concentration and C_labels is not None:
                trunk_conc = tf.concat(
                    [tf.zeros_like(tau_conc), tau_conc], axis=-1
                )
                _, _, C_pred = self.model(
                    [branch_conc, trunk_conc], B0_at_t=None
                )
                conc_loss = self.loss_fn(C_labels, C_pred)
            else:
                conc_loss = tf.constant(0.0)

            # ---- 质量守恒损失 ----
            if self.model.predict_concentration and branch_mass is not None:
                mass_loss = self._compute_mass_conservation_loss(
                    branch_mass, tau_mass,
                )
            else:
                mass_loss = tf.constant(0.0)

            total_loss = (
                self.lambda_data * data_loss
                + self.lambda_pde * pde_loss
                + self.lambda_flux * flux_loss
                + self.lambda_conc * conc_loss
                + self.lambda_mass * mass_loss
            )

        grads = outer_tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        return total_loss, data_loss, pde_loss, flux_loss, conc_loss, mass_loss

    def _val_step(self, branch_in, trunk_in, labels):
        n_pred, _, _ = self.model([branch_in, trunk_in], B0_at_t=None)
        return self.loss_fn(labels, n_pred)

    def _val_conc_step(self, branch_conc, tau_conc, C_labels):
        """验证集上的浓度损失。"""
        if not self.model.predict_concentration:
            return tf.constant(0.0)
        trunk_conc = tf.concat(
            [tf.zeros_like(tau_conc), tau_conc], axis=-1
        )
        _, _, C_pred = self.model(
            [branch_conc, trunk_conc], B0_at_t=None
        )
        return self.loss_fn(C_labels, C_pred)

    # ------------------------------------------------------------------
    # 配点数据生成
    # ------------------------------------------------------------------

    def generate_collocation(
        self,
        dataset,
        n_points_per_sheet: int = 2000,
        rng: Optional[np.random.Generator] = None,
    ):
        """为所有训练工况生成 PDE 配点数据。

        Returns
        -------
        branch_col  : (N, branch_dim)
        trunk_col   : (N, 2)
        G_tilde_col : (N, 1)
        B0_norm_col : (N, 1)
        """
        if rng is None:
            rng = np.random.default_rng(0)

        branch_list, trunk_list, G_list, B0_list = [], [], [], []

        for name in dataset.train_sheets:
            record = dataset._raw[name]
            branch_vec = dataset._build_branch_vector(record)

            xi = rng.uniform(0, 1, size=(n_points_per_sheet, 1))
            tau = rng.uniform(0.001, 1, size=(n_points_per_sheet, 1))

            t_phys = tau.flatten() * self.kinetics.t_max
            G_tilde = self.kinetics.G_normalized(name, t_phys).reshape(-1, 1)
            B0_flux = self.kinetics.B0_flux_normalized(name, t_phys).reshape(-1, 1)

            branch_list.append(np.tile(branch_vec, (n_points_per_sheet, 1)))
            trunk_list.append(np.hstack([xi, tau]))
            G_list.append(G_tilde)
            B0_list.append(B0_flux)

        return (
            np.concatenate(branch_list).astype(np.float32),
            np.concatenate(trunk_list).astype(np.float32),
            np.concatenate(G_list).astype(np.float32),
            np.concatenate(B0_list).astype(np.float32),
        )

    def generate_mass_collocation(
        self,
        dataset,
        n_points_per_sheet: int = 500,
        rng: Optional[np.random.Generator] = None,
    ):
        """为所有训练工况生成质量守恒配点数据。

        每个配点只需 (branch, τ)，积分在训练步内用 Gauss-Legendre 完成。

        Returns
        -------
        branch_mass : (N, branch_dim)
        tau_mass    : (N, 1)
        """
        if rng is None:
            rng = np.random.default_rng(1)

        branch_list, tau_list = [], []

        for name in dataset.train_sheets:
            record = dataset._raw[name]
            branch_vec = dataset._build_branch_vector(record)

            tau = rng.uniform(0.01, 1, size=(n_points_per_sheet, 1))

            branch_list.append(np.tile(branch_vec, (n_points_per_sheet, 1)))
            tau_list.append(tau)

        return (
            np.concatenate(branch_list).astype(np.float32),
            np.concatenate(tau_list).astype(np.float32),
        )

    # ------------------------------------------------------------------
    # 数据管道
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataset(arrays, batch_size, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(arrays)
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(arrays[0]), 100000))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # ------------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        colloc_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        conc_train_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        mass_colloc_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        conc_val_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        epochs: int = 300,
        batch_size: int = 4096,
        colloc_batch_size: int = 2048,
        conc_batch_size: int = 512,
        mass_batch_size: int = 256,
        print_every: int = 20,
        save_dir: Optional[str] = None,
        save_every: int = 50,
    ):
        train_ds = self._build_dataset(train_data, batch_size)
        colloc_ds = self._build_dataset(colloc_data, colloc_batch_size)

        conc_ds = (
            self._build_dataset(conc_train_data, conc_batch_size)
            if conc_train_data is not None
            else None
        )
        mass_ds = (
            self._build_dataset(mass_colloc_data, mass_batch_size)
            if mass_colloc_data is not None
            else None
        )
        val_ds = (
            self._build_dataset(val_data, batch_size, shuffle=False)
            if val_data is not None
            else None
        )
        val_conc_ds = (
            self._build_dataset(conc_val_data, conc_batch_size, shuffle=False)
            if conc_val_data is not None
            else None
        )

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            ep_total, ep_data, ep_pde, ep_flux = [], [], [], []
            ep_conc, ep_mass = [], []

            colloc_iter = iter(colloc_ds)
            conc_iter = iter(conc_ds) if conc_ds is not None else None
            mass_iter = iter(mass_ds) if mass_ds is not None else None

            for b_d, t_d, lbl in train_ds:
                # PDE 配点
                try:
                    b_c, t_c, g_c, b0_c = next(colloc_iter)
                except StopIteration:
                    colloc_iter = iter(colloc_ds)
                    b_c, t_c, g_c, b0_c = next(colloc_iter)

                # 浓度数据
                bc_conc, tau_c, C_lbl = None, None, None
                if conc_iter is not None:
                    try:
                        bc_conc, tau_c, C_lbl = next(conc_iter)
                    except StopIteration:
                        conc_iter = iter(conc_ds)
                        bc_conc, tau_c, C_lbl = next(conc_iter)

                # 质量守恒配点
                bm, tm = None, None
                if mass_iter is not None:
                    try:
                        bm, tm = next(mass_iter)
                    except StopIteration:
                        mass_iter = iter(mass_ds)
                        bm, tm = next(mass_iter)

                total, data_l, pde_l, flux_l, conc_l, mass_l = self._train_step(
                    b_d, t_d, lbl,
                    b_c, t_c, g_c, b0_c,
                    bc_conc, tau_c, C_lbl,
                    bm, tm,
                )
                ep_total.append(float(total))
                ep_data.append(float(data_l))
                ep_pde.append(float(pde_l))
                ep_flux.append(float(flux_l))
                ep_conc.append(float(conc_l))
                ep_mass.append(float(mass_l))

            self.history["train_total"].append(float(np.mean(ep_total)))
            self.history["train_data"].append(float(np.mean(ep_data)))
            self.history["train_pde"].append(float(np.mean(ep_pde)))
            self.history["train_flux"].append(float(np.mean(ep_flux)))
            self.history["train_conc"].append(float(np.mean(ep_conc)))
            self.history["train_mass"].append(float(np.mean(ep_mass)))

            val_loss = None
            if val_ds is not None:
                vl = []
                for b_v, t_v, lbl_v in val_ds:
                    vl.append(float(self._val_step(b_v, t_v, lbl_v)))
                val_loss = float(np.mean(vl))
                self.history["val_loss"].append(val_loss)

            val_conc_loss = None
            if val_conc_ds is not None:
                vcl = []
                for bc_v, tau_v, C_v in val_conc_ds:
                    vcl.append(float(self._val_conc_step(bc_v, tau_v, C_v)))
                val_conc_loss = float(np.mean(vcl))
                self.history["val_conc_loss"].append(val_conc_loss)

            if epoch % print_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                msg = (
                    f"Epoch {epoch:>4d}/{epochs} | "
                    f"total={np.mean(ep_total):.4e} "
                    f"data={np.mean(ep_data):.4e} "
                    f"pde={np.mean(ep_pde):.4e} "
                    f"flux={np.mean(ep_flux):.4e} "
                    f"conc={np.mean(ep_conc):.4e} "
                    f"mass={np.mean(ep_mass):.4e}"
                )
                if val_loss is not None:
                    msg += f" | val={val_loss:.4e}"
                if val_conc_loss is not None:
                    msg += f" valC={val_conc_loss:.4e}"
                msg += f" | {elapsed:.2f}s"
                print(msg)

            if save_dir and epoch % save_every == 0:
                ckpt_path = os.path.join(save_dir, f"ckpt_epoch_{epoch:04d}", "pi_deeponet")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                self.model.save_weights(ckpt_path)
                print(f"  [Checkpoint] saved: {ckpt_path}")
