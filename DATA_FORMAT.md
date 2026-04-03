# 数据格式说明：Excel → Pickle 缓存对照

## 一、总体流程

```
Excel (.xlsx)                        Pickle 缓存 (.pkl)
┌──────────────────┐     首次运行      ┌──────────────────────┐
│ 25 个 Sheet      │  ──自动转换──→   │ Python dict          │
│ 每个 Sheet =     │     ~8 min       │ {sheet_name: record} │
│ 一个实验工况     │                  │ 加载 < 1 秒          │
└──────────────────┘                  └──────────────────────┘
        ↑                                      ↑
  原始仿真数据                          程序实际使用的格式
  (人工准备)                          (自动生成, 无需手动维护)
```

**你只需准备 Excel 文件，程序会自动在同目录生成缓存。**
**如果修改了 Excel，程序会检测到时间戳变化并重新生成缓存。**

---

## 二、Excel 文件结构要求

文件名: 任意 `.xlsx`，当前为 `Simulation_Results_Parallel.xlsx`

### Sheet 命名

每个 Sheet = 一个实验工况，命名如 `CR_1_00`, `CR_2_50` 等。
所有 Sheet 结构必须完全一致（相同列名、相同 L 网格、相同快照时刻数）。

### 列结构（每个 Sheet 必须包含）

```
列名                  含义                        长度          示例值
─────────────────────────────────────────────────────────────────────────
Time_s               仿真时间序列 (秒)           N_time        0, 0.1, 0.2, ..., 10799.9
Temp_K               温度随时间变化 (K)          N_time        333.15, 333.15, ...
Conc                 浓度随时间变化              N_time        0.048187, 0.048186, ...
L_mid_um             粒径网格中心点 (μm)         N_L           0.5, 1.5, 2.5, ..., 2999.5
Growth_Rate_G        生长速率 G(t) (μm/s)        N_time        0.019, 0.020, ...
Nuc_Rate_B           成核速率 B₀(t)              N_time        2e-11, 3e-11, ...
Time_0s              PSD 快照: t=0s 时的 n(L)    N_L           0, 0, 2e-12, ...
Time_180s            PSD 快照: t=180s 时的 n(L)  N_L           0, 0, 1e-10, ...
Time_360s            PSD 快照: t=360s            N_L           ...
...                  (每隔 180s 一个快照)         N_L           ...
Time_10800s          PSD 快照: t=10800s          N_L           ...
```

### 当前数据的具体数值

| 参数 | 值 |
|------|------|
| N_time (时间步数) | 108,000 (dt=0.1s, 总 10800s) |
| N_L (粒径 bin 数) | 3,000 (0.5~2999.5 μm) |
| 快照数 | 61 个 (t=0, 180, 360, ..., 10800s) |
| Sheet 数 | 25 个工况 |

### PSD 快照列的命名规则

```
列名格式:  Time_{秒数}s
例如:      Time_0s, Time_180s, Time_360s, ..., Time_10800s
```

**注意:** `Time_s` (时间序列列) 和 `Time_0s` (PSD 快照列) 是两个不同的列！
程序通过 `startswith("Time_") and != "Time_s"` 来区分。

---

## 三、Excel → 内部数据的转换规则

每个 Sheet 被解析为以下 Python 字典 (`record`)：

```
Excel 列                    →  record 字段              →  shape / type
──────────────────────────────────────────────────────────────────────────
Time_s                      →  record["Time_s"]         →  ndarray (N_time,)
Temp_K                      →  record["Temp_K"]         →  ndarray (N_time,)
Conc                        →  record["Conc"]           →  ndarray (N_time,)
L_mid_um                    →  record["L_mid_um"]       →  ndarray (N_L,)
Growth_Rate_G               →  record["Growth_Rate_G"]  →  ndarray (N_time,)  [PI-DeepONet 用]
Nuc_Rate_B                  →  record["Nuc_Rate_B"]     →  ndarray (N_time,)  [PI-DeepONet 用]

Time_0s, Time_180s, ...     →  record["snapshot_times"] →  ndarray (61,) = [0, 180, 360, ...]
 (列名中提取秒数)              record["psd"]            →  ndarray (61, 3000)
                                                            psd[i, j] = 第 i 个快照, 第 j 个 L bin 的 n(L,t)

Conc[0]                     →  record["C0"]             →  float (初始浓度)
psd[0]                      →  record["n_L0"]           →  ndarray (N_L,) (初始 PSD)
```

---

## 四、Pickle 缓存文件

**文件名:** `{Excel文件名去掉.xlsx}_cache.pkl`

例如: `Simulation_Results_Parallel.xlsx` → `Simulation_Results_Parallel_cache.pkl`

**内容:** Python 字典

```python
{
    "CR_1_00": {
        "Time_s":         np.ndarray,  # (108000,)
        "Temp_K":         np.ndarray,  # (108000,)
        "Conc":           np.ndarray,  # (108000,)
        "L_mid_um":       np.ndarray,  # (3000,)
        "snapshot_times": np.ndarray,  # (61,)
        "psd":            np.ndarray,  # (61, 3000)
        "Growth_Rate_G":  np.ndarray,  # (108000,)
        "Nuc_Rate_B":     np.ndarray,  # (108000,)
        "C0":             float,
        "n_L0":           np.ndarray,  # (3000,)
    },
    "CR_1_12": { ... },
    ...
}
```

---

## 五、准备新数据集的检查清单

1. **所有 Sheet 的 L_mid_um 列必须完全相同** (相同 bin 数和值)
2. **所有 Sheet 的快照时刻必须完全相同** (相同的 Time_xxxs 列名)
3. **Time_s, Temp_K, Conc 长度必须一致** (同一 dt)
4. **Growth_Rate_G 和 Nuc_Rate_B 列可选**:
   - Data-Driven (`train.py`) 不需要这两列
   - PI-DeepONet (`train_pi.py`) 需要这两列来计算物理残差
5. **PSD 快照列中前 N_L 行是有效数据**, 超出部分会被忽略
6. **将新 Excel 放到 `learn/` 目录下**, 修改 `train.py` / `train_pi.py` 中的 `EXCEL_PATH`
7. **删除旧缓存** `*_cache.pkl`, 程序会自动重建

---

## 六、数据在 DeepONet 中的使用方式

```
               Branch 输入 (162 维)                    Trunk 输入 (2 维)
┌──────────────────────────────────────────┐    ┌──────────────────┐
│ T(t₀) T(t₁) ... T(t₆₀)  │ n(L,0)下采样 │ C₀ │    │  L_norm   t_norm  │
│      61 个温度值          │   100 个值   │ 1  │    │  L/L_max  t/t_max │
└──────────────────────────────────────────┘    └──────────────────┘
         ↓                                              ↓
    Branch Net (MLP)                              Trunk Net (MLP)
         ↓                                              ↓
     (batch, 128)                                  (batch, 128)
         └─────────── 逐元素乘 + 求和 + bias ──────────┘
                              ↓
                     n(L, t) 预测值 (归一化)
```

### 归一化规则

| 量 | 归一化方式 | 说明 |
|----|-----------|------|
| T(t) | (T - T_min) / (T_max - T_min) | 全局 min-max |
| C(0) | (C - C_min) / (C_max - C_min) | 全局 min-max |
| L | L / L_max | 除以最大粒径 |
| t | t / t_max | 除以最大时间 |
| n(L,t) | n / n_scale | n_scale = 所有正值 PSD 的第 99 百分位 |
