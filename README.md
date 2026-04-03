# PBE_DeepONet — 结晶过程粒数衡算方程的算子学习

使用 Physics-Informed Deep Operator Network (PI-DeepONet) 学习结晶过程中 **PBE (Population Balance Equation)** 的算子映射：

**给定初始条件 n(L,0)、C(0) 和温度曲线 T(t)，预测任意时刻的粒度分布 n(L,t) 和溶质浓度 C(t)，并满足质量守恒定律。**

## 项目结构

```
PBE_DeepONet/
├── train.py                   # 纯数据驱动 DeepONet 训练脚本
├── train_pi.py                # PI-DeepONet 训练脚本（含物理约束 + 质量守恒）
├── predict.py                 # 预测脚本（加载权重、PSD 推理与可视化）
├── requirements.txt           # Python 依赖
├── README.md
├── DATA_FORMAT.md             # Excel 数据列格式详细说明
├── results/                   # data-driven 训练输出
├── results_pi/                # PI-DeepONet 训练输出
├── predictions/               # 预测输出
└── deeponet_pbe/              # 核心包
    ├── __init__.py
    ├── data.py                # 数据加载与预处理（含浓度数据构建）
    ├── model.py               # 标准 DeepONet 模型
    ├── pi_model.py            # PI-DeepONet 模型（硬约束 + 浓度预测）
    ├── trainer.py             # 标准训练循环（仅 MSE）
    ├── pi_trainer.py          # PI 训练循环（PDE + 通量 + 浓度 + 质量守恒）
    ├── kinetics.py            # 结晶动力学模块（G, B₀, C 插值与 α 估计）
    ├── utils.py               # 可视化工具（PSD、浓度对比图）
    └── gpu_config.py          # GPU 配置
```

## 物理背景

### 粒数衡算方程 (PBE)

一维结晶过程（无生长弥散 Dg=0）：

$$\frac{\partial n}{\partial t} + G(t) \frac{\partial n}{\partial L} = 0$$

- n(L,t)：粒度分布密度
- G(t)：晶体生长速率
- 边界条件：J(L=0, t) = B₀(t)（成核速率）

### 质量守恒

溶质浓度与晶体质量的耦合关系：

$$C(t) = C(0) - \rho_c \cdot k_v \cdot \mu_3(t), \quad \mu_3(t) = \int_0^{L_{max}} L^3 \, n(L,t) \, dL$$

归一化后：

$$C_{norm}(\tau) + \alpha_{norm} \cdot \int_0^1 \xi^3 \, n_{norm}(\xi, \tau) \, d\xi = C_0^{norm}$$

其中 α_norm 从仿真数据通过加权最小二乘法自动估计。

## 两种训练模式

### 1. Data-Driven DeepONet (`train.py`)

纯数据监督，仅 MSE 损失。

### 2. PI-DeepONet (`train_pi.py`) — 推荐

包含 5 项物理约束损失：

| 损失项 | 公式 | 说明 |
|--------|------|------|
| L_data | MSE(n_pred, n_true) | PSD 数据监督 |
| L_pde | ∂n/∂τ + G̃·∂n/∂ξ = 0 | PDE 残差（自动微分） |
| L_flux | J - G̃·n = 0 | 通量本构残差 |
| L_conc | MSE(C_pred, C_true) | 浓度数据监督 |
| L_mass | C + α·μ₃ - C₀ = 0 | 质量守恒残差（Gauss-Legendre 积分） |

$$\mathcal{L} = \lambda_{data} \mathcal{L}_{data} + \lambda_{pde} \mathcal{L}_{pde} + \lambda_{flux} \mathcal{L}_{flux} + \lambda_{conc} \mathcal{L}_{conc} + \lambda_{mass} \mathcal{L}_{mass}$$

## 硬约束

模型通过解析结构自动满足初始条件和边界条件（无需额外损失项）：

| 约束 | 实现方式 | 物理含义 |
|------|----------|----------|
| n(ξ, 0) = n₀(ξ) | n = n₀_interp(ξ) + τ·n_raw | 初始 PSD（从 branch 输入线性插值） |
| C(0) = C₀ | C = C₀_norm + τ·C_raw | 初始浓度 |
| J(0, t) = B₀(t) | J = φ(ξ)·B₀ + ξ(1-ξ)·J_raw | 成核通量边界 |
| J(L_max, t) = 0 | φ(ξ) = (1-ξ)·exp(-ξ/ε) | 大粒径端无通量 |

## 网络架构

```
                        PI-DeepONet 架构

Branch Net                        PSD Trunk Net
输入: [T传感器, n₀传感器, C₀]       输入: [ξ, τ]
     (batch, 162)                       (batch, 2)
         ↓                                 ↓
   MLP [256,256,256]                MLP [128,128,128]
         ↓                                 ↓
      b (batch, 128)                 t (batch, 128)
         ↓_____________↓________________↓
         |        b ⊙ t (逐元素积)       |
         |             ↓                  |
         |      ┌──────┴──────┐           |
         |   readout_n     readout_J      |
         |      ↓              ↓          |
         |   n_raw(ξ,τ)    J_raw(ξ,τ)    |
         |      ↓              ↓          |
         |   硬约束变换      硬约束变换    |
         |      ↓              ↓          |
         |   n(ξ,τ)        J(ξ,τ)        |
         |                                |
         ↓                                |
   Conc Trunk Net                         |
   输入: [τ]                              |
        ↓                                 |
   MLP [64,64]                            |
        ↓                                 |
   c_t (batch, 128)                       |
        ↓                                 |
     b ⊙ c_t → readout_C → C_raw         |
                    ↓                     |
              C₀ + τ·C_raw = C(τ)         |
```

### Branch 输入构成

| 部分 | 维度 | 说明 |
|------|------|------|
| T(t) 传感器 | 61 | 温度曲线在 PSD 快照时刻（0, 180, ..., 10800s）的采样值 |
| n(L,0) 传感器 | 100 | 初始粒度分布在 100 个等距 L 点的下采样值 |
| C(0) | 1 | 初始浓度 |
| **总计** | **162** | |

## 数据说明

数据源：`../learn/Simulation_Results_Parallel.xlsx`（详见 `DATA_FORMAT.md`）

- **25 个工况**：不同冷却速率（CR_1_00 ~ CR_4_00，对应 -4 ~ -16 K/h）
- **每个工况包含**：
  - 时间序列（dt=0.1s, 共 10800s）：温度 T(t)、浓度 C(t)、成核速率 B₀(t)、生长速率 G(t)
  - 粒度分布 PSD：3000 个 L bins (0.5~2999.5 μm) × 61 个时间快照（每 180s）

## 快速开始

```bash
cd PBE_DeepONet

# PI-DeepONet 训练（推荐）
python -u train_pi.py

# 纯数据驱动训练
python -u train.py

# 预测
python predict.py --sheet CR_2_50 --times 1800 3600 5400 7200
```

训练完成后 `results_pi/` 文件夹将包含：
- `pi_loss_curves.png` — 6 项损失曲线（2×3 面板）
- `pi_psd_*.png` — 各测试工况 PSD 对比
- `pi_evolution_*.png` — PSD 时间演化
- `pi_concentration_*.png` — 浓度预测 vs 仿真真值（含相对误差）

## 超参数

在 `train_pi.py` 中集中定义：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_L_SENSORS` | 100 | 初始 PSD 下采样数（Branch 输入） |
| `N_L_EVAL` | 200 | L 轴评估点数（Trunk 输入） |
| `BRANCH_HIDDENS` | [256,256,256] | Branch 网络隐藏层 |
| `TRUNK_HIDDENS` | [128,128,128] | PSD Trunk 网络隐藏层 |
| `CONC_TRUNK_HIDDENS` | [64,64] | 浓度 Trunk 网络隐藏层 |
| `LATENT_DIM` | 128 | Branch/Trunk 共同输出维度 |
| `EPOCHS` | 300 | 训练轮数 |
| `BATCH_SIZE` | 4096 | 数据批大小 |
| `LAMBDA_DATA` | 1.0 | PSD 数据损失权重 |
| `LAMBDA_PDE` | 0.1 | PDE 残差权重 |
| `LAMBDA_FLUX` | 0.1 | 通量本构残差权重 |
| `LAMBDA_CONC` | 1.0 | 浓度数据损失权重 |
| `LAMBDA_MASS` | 0.1 | 质量守恒残差权重 |
| `N_QUAD` | 32 | Gauss-Legendre 积分节点数 |
| `TEST_SHEETS` | CR_1/2/3/4_00 | 测试工况 |

## 数据归一化策略

| 量 | 归一化方式 | 说明 |
|----|-----------|------|
| 温度 T | (T - T_min) / (T_max - T_min) | 全局 min-max |
| 浓度 C | (C - C_min) / (C_max - C_min) | 全局 min-max（含全时间序列） |
| 粒径 L | L / L_max | 除以最大粒径 |
| 时间 t | t / t_max | 除以最大时间 |
| PSD n(L,t) | n / n_scale | n_scale = 所有正值 PSD 的第 99 百分位 |

## 参考文献

- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators*. Nature Machine Intelligence, 3(3), 218-229.
- Ramkrishna, D. (2000). *Population Balances: Theory and Applications to Particulate Systems in Engineering*. Academic Press.
