# 项目使用说明

## 项目概述

本仓库复现 Ferreira, Simchi-Levi & Wang (2015) 论文 **"Online Network Revenue Management using Thompson Sampling"**。核心方法是将 Thompson Sampling 与线性规划 (LP) 子程序结合，解决带库存约束的动态定价问题。

---

## 一、项目文件结构

```
project/
│
│   main.py              # 实验主入口（运行所有实验）
│   config.py            # 全局配置
│   requirements.txt     # Python 依赖
│   test_imports.py      # 导入测试
│   test_smoke.py        # 冒烟测试
│   benchmark.py         # 基准测试（测量本机仿真速度）
│
├── algorithms/          # 算法实现
│   ├── base.py                 # 算法基类
│   ├── ts_fixed.py             # TS-fixed (Algorithm 1)
│   ├── ts_update.py            # TS-update (Algorithm 2)
│   ├── ts_unconstrained.py     # 经典 Thompson Sampling（无约束基准）
│   ├── bz.py                   # Besbes & Zeevi (2012) 基准
│   ├── pd_bwk.py               # PD-BwK 基准 (UCB + Primal-Dual)
│   ├── ts_linear.py            # TS-linear 扩展（连续价格）
│   ├── ts_contextual.py        # TS-contextual 扩展（情境化）
│   └── ts_bwk.py               # TS-BwK 扩展（通用 BwK）
│
├── models/              # 环境模型
│   ├── demand.py               # 需求分布（伯努利、泊松、线性/指数/Logit）
│   ├── inventory.py            # 库存管理
│   ├── revenue_network.py      # 网络收益管理环境
│   └── posterior.py            # 后验分布更新（Beta, Gamma）
│
├── experiments/         # 实验配置与运行器
│   ├── config_single.py        # 单产品实验参数
│   ├── config_multi.py         # 多产品实验参数
│   └── runner.py               # 实验运行器（支持并行化）
│
├── utils/               # 工具
│   ├── lp_solver.py            # LP 求解封装（带缓存优化）
│   └── statistics.py           # 统计工具
│
├── analysis/            # 分析与可视化
│   ├── baseline.py             # LP(d) 最优基准计算
│   ├── regret.py               # Bayesian Regret 计算
│   ├── visualizer.py           # 画图工具
│   └── metrics.py              # 性能指标
│
├── figures/             # 输出图表（运行后生成）
│   ├── figure1_single_product.png   # 图1: 单产品结果
│   ├── figure2_multi_product.png    # 图2: 多产品结果
│   ├── figure3_regret_curve.png     # 图3: Regret 曲线
│   ├── figure4_heatmap.png          # 图4: 热力图
│   └── figure5_delta_performance.png # 图5: Δ性能对比
│
└── results/             # 实验数据（Pickle 格式）
    ├── single_alpha0.25.pkl   # α=0.25 单产品结果
    ├── single_alpha0.5.pkl    # α=0.5  单产品结果
    ├── multi_linear.pkl       # 线性需求多产品结果
    ├── multi_exponential.pkl  # 指数需求多产品结果
    └── multi_logit.pkl        # Logit 需求多产品结果
```

---

## 二、预期输出文件

### 2.1 results/ 目录（Pickle 数据文件）

| 文件名 | 来源 | 内容 |
|--------|------|------|
| `single_alpha0.25.pkl` | 单产品 α=0.25 | 5组 T 值 × 5种算法的收入与Regret |
| `single_alpha0.5.pkl` | 单产品 α=0.5 | 同上 |
| `multi_linear.pkl` | 多产品-线性需求 | 5组 T 值 × 3种算法 |
| `multi_exponential.pkl` | 多产品-指数需求 | 同上 |
| `multi_logit.pkl` | 多产品-Logit需求 | 同上 |

每个 `.pkl` 文件存储一个字典，结构如下：
```python
{
  "T=500_inv=0.25_demand=bernoulli": {
    "baseline": 0.0,          # LP(d) 最优值
    "algorithms": {
      "TS-fixed": {
        "mean_revenue": 0.0,
        "std_revenue": 0.0,
        "mean_pct_optimal": 0.0,
        "per_period_revenues": [[...], ...]  # n_sim × T 矩阵
      },
      "TS-update": {...},
      "BZ": {...},
      ...
    }
  },
  ...
}
```

### 2.2 figures/ 目录（图表）

| 图表 | 对应论文 | 内容 | 生成命令 |
|------|---------|------|---------|
| `figure1_single_product.png` | Figure 1 | 单产品：% of optimal revenue vs. T，α∈{0.25, 0.5} | `--experiment single` |
| `figure2_multi_product.png` | Figure 2 | 多产品：% of optimal revenue vs. T，三种需求函数 | `--experiment multi` |
| `figure3_regret_curve.png` | 新增 | Regret vs. T（对数坐标），展示 O(√T) 量级 | `--experiment extended` |
| `figure4_heatmap.png` | 新增 | 热力图：α × T × 最优收入百分比 | 同上 |
| `figure5_delta_performance.png` | 新增 | Δ 性能 (TS-update − TS-fixed) vs. α | 同上 |

### 2.3 终端输出（运行时打印）

| 输出内容 | 说明 |
|---------|------|
| 每个 α / 需求类型的实验进度条 | 显示当前 T 配置完成进度 |
| 汇总统计表 | 每种算法在 T=10000 时的均值、标准差、最优收入百分比 |
| 耗时信息 | 每个实验块的完成时间 |

---

## 三、运行命令

### 3.1 基础命令

```bash
# 安装依赖（首次运行前）
pip install -r requirements.txt

# 验证安装
python test_imports.py && python test_smoke.py
```

### 3.2 实验命令及耗时预估

项目集成了两项优化：
1. **P0 — 并行化**：`runner.py` 使用 `joblib.Parallel` 自动利用全部 CPU 核心（`n_jobs=-1`），仿真间完全独立
2. **P1 — LP 求解优化**：`lp_solver.py` 缓存 bounds 列表 + 预分配数组，单次仿真加速约 4x

**耗时方法说明**：benchmark.py 先实测 T=1000 和 T=10000 的 per-sim 时间，再用幂律插值（时间 ∝ T^b，当前 b≈1.35）分段估算各 T 消耗，避免了早期版本用单一 per-sim 线性外推导致的 5× 低估。

以下耗时基于 AMD Ryzen 9 7945HX (16核/32线程) 实测，含 LP 求解超线性缩放修正：

| 命令 | 说明 | 串行耗时 | 并行耗时 (~14×) |
|------|------|---------|----------------|
| `python main.py --experiment single --n_sim 500` | 复现 Figure 1（全量） | **~7.3 小时** | **~32 分钟** |
| `python main.py --experiment single --n_sim 50` | 快速验证单产品 | **~44 分钟** | **~3 分钟** |
| `python main.py --experiment multi --n_sim 100` | 复现 Figure 2 | **~45 分钟** | **~3 分钟** |
| `python main.py --experiment multi --n_sim 50` | 快速验证多产品 | **~23 分钟** | **~2 分钟** |
| `python main.py --experiment extended --n_sim 50` | 图3-5 + 表1 | **~24 分钟** | **~2 分钟** |
| `python main.py --all --n_sim 500` | 运行全部实验（全量） | **~8.5 小时** | **~37 分钟** |

> **注意**：并行模式自动启用。如果希望降级为串行（如调试时），设置环境变量 `JOBS=1` 或在 `runner.py` 中传入 `n_jobs=1`。

> **为什么单产品实验耗时远大于多产品？** 单产品实验有 5 种算法 × 500 次仿真 = 25,000 次仿真；多产品实验仅 3 种算法 × 100 次仿真 = 4,500 次仿真。且多产品 per-sim 实际更快（Poisson+Gamma 后验 + 多资源 LP 在 HiGHS 求解器中收敛更快，实测 per-sim 仅为单产品的 0.57×）。此外 LP 求解时间随 T 超线性增长（T=1000→T=10000 放大 ~22.6×），并非线性缩放，因此不能直接用 T=1000 的 per-sim 时间简单乘以 config 数量。

### 3.3 仅绘图（重新渲染已有结果）

如果想跳过仿真、直接基于已有 `.pkl` 数据重新画图，可修改 `main.py` 中的对应部分，或单独调用 `analysis/visualizer.py` 中的函数。

### 3.4 基准测试

```bash
# 运行基准测试，测量本机仿真速度并外推全量耗时
python benchmark.py
```

### 3.5 测试命令

```bash
# 导入测试（验证所有模块可正确导入）
python test_imports.py

# 冒烟测试（运行极简版实验，检查有无报错）
python test_smoke.py
```

---

## 四、实验参数速查

### 单产品实验参数

| 参数 | 值 |
|------|-----|
| 产品数 N | 1 |
| 资源数 M | 1 |
| 价格数 K | 4 |
| 价格集合 | [$29.90, $34.90, $39.90, $44.90] |
| 均值需求 | [0.8, 0.6, 0.3, 0.1] |
| 需求分布 | Bernoulli |
| 先验 | Beta(1, 1) |
| T 取值 | [500, 1000, 2000, 5000, 10000] |
| α 取值 | [0.25, 0.5] |

### 多产品实验参数

| 参数 | 值 |
|------|-----|
| 产品数 N | 2 |
| 资源数 M | 3 |
| 价格数 K | 5 |
| 需求分布 | Poisson |
| 先验 | Gamma(1, 1) |
| T 取值 | [500, 1000, 2000, 5000, 10000] |
| 需求类型 | linear, exponential, logit |

---

## 五、算法对照表

| 算法 | 类名 | 来源 | 关键特征 |
|------|------|------|---------|
| TS-fixed | `TSFixed` | 论文 Algorithm 1 | 固定库存率 c_j = I_j / T |
| TS-update | `TSUpdate` | 论文 Algorithm 2 | 动态库存率 c_j(t) |
| BZ | `BZAlgorithm` | Besbes & Zeevi (2012) | 探索-利用分离，τ = T^{2/3} |
| PD-BwK | `PDBwK` | Badanidiyuru et al. (2013) | UCB + Primal-Dual |
| TS (unconstrained) | `TSUnconstrained` | 经典 TS | 无库存约束，不作为收敛基准 |
| TS-linear | `TSLinear` | 论文 Section 4.1 | 连续价格 + 线性需求 |
| TS-contextual | `TSContextual` | 论文 Section 4.2 | 情境化特征（如会员/非会员） |
| TS-BwK | `TSBwK` | 论文 Section 4.3 | 通用 Bandit with Knapsacks |

---

## 六、常见问题

**Q: 仿真太慢怎么办？**
A: 全量实验在 16 核 AMD Ryzen 9 7945HX 上并行约 37 分钟即可完成（串行约 8.5 小时）。如果还需要更快的迭代，先用 `--n_sim 10` 验证逻辑（单产品约 1 分钟并行），再用 `--n_sim 500` 跑最终结果。

**Q: LP 求解出现 `OptimizeWarning`（tol 参数）？**
A: 这是 scipy 版本兼容性问题（HiGHS 求解器在部分 scipy 版本中不识别 `tol` 参数），不影响结果的正确性，可忽略。

**Q: 如何中断后继续？**
A: 结果在每个 α/需求类型完成时自动保存为 `.pkl` 文件。若某一步中断，可直接修改 `main.py` 跳过已完成步骤，或单独运行对应的函数。

**Q: 并行化不生效？**
A: 确保已安装 joblib（`pip install joblib`）。在 Windows 下，`loky` 后端自动处理进程启动，无需额外配置。

**Q: 串行模式下结果与并行模式一致吗？**
A: 是的。并行化仅改变执行顺序，每个仿真使用不同随机种子，且 env_seed 和 algo_seed 的分隔方式保证了结果的独立同分布性质。

---

## 七、基准测试数据

以下数据通过运行 `python benchmark.py` 在 **AMD Ryzen 9 7945HX** 上实测获得。

### 硬件环境

| 项目 | 值 |
|------|-----|
| 操作系统 | Windows 11 |
| CPU | AMD Ryzen 9 7945HX (16核 / 32线程) |
| Python 环境 | Python 3.10.9 |
| 并发模式 | 串行基准测试（用于推算并行加速） |

### 优化效果对比

| 优化阶段 | T=1000 单次仿真 | T=10000 单次仿真 | 加速比 |
|---------|----------------|-----------------|--------|
| 原始版本（无优化） | 0.67s | 10.76s | 1x (baseline) |
| 当前版本（P1 缓存+预分配） | **0.149s** | **3.36s** | **~3-4x** |
| 并行化（P0, 16核） | **~0.01s** 有效 | — | **~14x 额外** |

### 串行基准测试结果

| 测试项目 | 配置 | 结果 |
|---------|------|------|
| T=1000, 10sims × 5算法 | 单产品, α=0.25 | **7.4s 总计** (0.149s/次) |
| T=10000, 3sims × 1算法 | 单产品, α=0.25, TS-fixed | **10.1s 总计** (3.36s/次) |
| T=1000, 10sims × 3算法 | 多产品, linear 需求 | **2.5s 总计** (0.085s/次, 多/单=0.57×) |
| T=1000 → T=10000 缩放因子 | — | **22.6×** (LP 求解超线性缩放，指数 b≈1.35) |

### T 分段外推全量实验时间（串行 vs 并行）

以下外推使用幂律插值（时间 ∝ T^b），按每个 T 值分别估算 per-sim 时间后求和，避免早期版本用单一 per-sim 线性外推导致的 5× 低估。

**单产品实验 (5T × 2α × 500sims × 5algos = 25,000 sims)：**

| T | sims | per-sim | 串行时间 |
|---|------|---------|----------|
| 500 | 5,000 | 0.058s | 4.8 min |
| 1000 | 5,000 | 0.149s | 12.4 min |
| 2000 | 5,000 | 0.380s | 31.7 min |
| 5000 | 5,000 | 1.316s | 109.6 min |
| 10000 | 5,000 | 3.364s | 280.3 min |
| **合计** | 25,000 | — | **438.9 min = 7.3 hr** |

| 实验 | 总仿真次数 | 串行时间 | 并行时间 (~14×) |
|------|-----------|---------|----------------|
| 单产品 (5T × 2α × 500sims × 5algos) | 25,000 | **7.31 hr** | **~32 min** |
| 多产品 (5T × 3demand × 100sims × 3algos) | 4,500 | **0.75 hr** | **~3 min** |
| 扩展实验 (Fig 3-5 + Table 1) | ~1,575 | **0.40 hr** | **~2 min** |
| **全量总计** | ~31,075 | **8.46 hr** | **~37 min** |

### 注意

- **串行→并行的加速比约为 14×**（16 核 × 85% 实际效率），略低于理论值因 Windows 进程启动开销（~0.3-0.5s/进程）。大 T 仿真（per-sim > 3s）时加速比更接近 16×。
- `joblib` 使用 `loky` 后端自动处理 Windows 下的进程管理，无需手动配置。
- **LP 求解时间随 T 超线性增长**（指数 b≈1.35），T=10000 的 per-sim 时间是 T=1000 的 22.6 倍，而非直观的 10 倍。这是 HiGHS 求解器在资源约束 LP 上的实际复杂度表现，因此不能用 T=1000 的 per-sim 简单线性外推。
- 多产品实验 per-sim 反而快于单产品（0.57×）：Poisson+Gamma 后验的 LP 在 HiGHS 中收敛路径更短，具体原因与数值条件有关。扩展实验的 config 数量基于 main.py 中实际定义：Fig 3 的 1 config、Fig 4 的 36 configs、Fig 5 的 9 configs、Table 1 的 2 configs。
