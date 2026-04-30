# Online Network Revenue Management using Thompson Sampling — 复现行动方案

---

## 一、问题背景与动机

### 1.1 问题描述

一个零售商在有限销售季节内，用**有限库存**销售**多种产品**，需要通过动态调整价格来最大化总收入。具体来说：

- **资源约束**：$M$ 种资源，每种资源 $j$ 有初始库存 $I_j$，销售期内不可补货
- **产品与资源关系**：每单位产品 $i$ 消耗 $a_{ij}$ 单位资源 $j$
- **价格决策**：每个周期 $t = 1,\dots,T$，零售商从 $K$ 个价格向量 $\{p_1,\dots,p_K\}$ 中选择一个
- **需求未知**：给定价格 $p_k$，需求 $D(t)$ 服从参数 $\theta$ 的分布 $F(x; p_k, \theta)$，$\theta$ 未知
- **目标**：在 $T$ 个周期内最大化期望总收入

### 1.2 核心挑战：探索-利用权衡

零售商面临两个相互冲突的目标：

| 目标 | 内容 | 代价 |
|------|------|------|
| **探索 (Exploration)** | 尝试不同价格以学习需求参数 $\theta$ | 现在可能定价不优，且消耗了本可用于利用的库存 |
| **利用 (Exploitation)** | 基于已知信息选择收入最高的价格 | 若误判需求，会永远锁定次优价格 |

额外的**库存约束**让问题更难：探索时用掉的库存无法再用于利用阶段。

### 1.3 为什么需要混合策略？

论文的一个关键洞察是：当存在库存约束时，最优策略通常不是单一价格，而是**多个价格的混合分布**。

以论文单产品为例：
- 库存 $I = 0.25T$ 时，最优策略是：$3/4$ 的顾客给 $39.90$，$1/4$ 的顾客给 $44.90$（论文 Figure 1 说明）
- 没有库存约束时（经典 MAB），最优策略是单一最高收入价格 $29.90$（但这在有库存约束时是次优的）

这就是为什么不能直接套用经典 Thompson Sampling，而需要引入 LP 子程序。

### 1.4 现有方法及其局限

| 方法 | 代表文献 | 局限 |
|------|---------|------|
| 探索-利用分离法 | Besbes & Zeevi (2012) | 探索阶段后不再更新需求估计，且探索阶段可能耗尽库存 |
| UCB + Primal-Dual | Badanidiyuru et al. (2013) | 学习每个价格独立，无法利用价格间的相关性 |
| 贪婪贝叶斯法 | Harrison et al. (2012) | 不主动探索，可能陷入次优 |
| **本论文：TS + LP** | **Ferreira et al. (2015)** | **同时解决探索-利用和库存约束，后验采样自然实现探索** |

---

## 二、核心思想与算法

### 2.1 总体框架

论文的核心创新是将 Thompson Sampling 与线性规划 (LP) 子程序结合，形成**两步法**：

```
每周期 t：
  Step 1（TS 部分）：从后验分布 P(θ | H_{t-1}) 采样参数 θ(t)
  Step 2（LP 部分）：求解 LP(d(t)) 得到价格混合策略 x(t)
  Step 3（执行）：按 x(t) 概率选择价格 p_k
  Step 4（更新）：观察到需求 D(t)，更新后验分布
```

### 2.2 模型形式化

**符号表：**

| 符号 | 含义 |
|------|------|
| $N$ | 产品数量 |
| $M$ | 资源数量 |
| $K$ | 可用的价格向量数量 |
| $T$ | 销售季节长度（周期数） |
| $I_j$ | 资源 $j$ 的初始库存 |
| $a_{ij}$ | 单位产品 $i$ 消耗的资源 $j$ 数量 |
| $p_k$ | 第 $k$ 个价格向量（长度为 $N$） |
| $d_{ik}(t)$ | 采样参数下产品 $i$ 在价格 $k$ 的期望需求 |
| $c_j = I_j/T$ | 资源 $j$ 平均每周期可用库存 |
| $x_k(t)$ | 周期 $t$ 选择价格 $k$ 的概率 |

### 2.3 算法 1：TS-fixed（固定库存约束）

```
Algorithm 1: Thompson Sampling with Fixed Inventory Constraints (TS-fixed)

对于每个周期 t = 1, ..., T:
  1. 从后验分布采样 θ(t)，计算均值需求 d(t) = {d_{ik}(t)}

  2. 求解线性规划 LP(d(t)):
     目标:  max  Σ_k (Σ_i p_{ik} d_{ik}(t)) x_k
     约束:  Σ_k (Σ_i a_{ij} d_{ik}(t)) x_k ≤ c_j,  ∀j
            Σ_k x_k ≤ 1
            x_k ≥ 0, ∀k

  3. 以概率 x_k(t) 选择价格 p_k，以 1-Σx_k 选择"关闭价格" p_∞

  4. 观察需求 D(t)，更新后验分布
```

**LP 的直观含义：**
- 目标函数：给定采样参数下的期望每周期收入
- 第一个约束：期望资源消耗不超过平均可用库存
- 第二个约束：概率和为 1

### 2.4 算法 2：TS-update（动态库存约束）

与 TS-fixed 的唯一区别：将固定常数 $c_j$ 替换为实时更新的 $c_j(t)$：

$$c_j(t) = \frac{I_j(t-1)}{T - t + 1}$$

即：剩余库存除以剩余周期数。这个看似微小的变化让算法能**感知库存耗尽风险**，从而更及时地调整定价策略。

### 2.5 三种扩展算法

| 算法 | 适用场景 | 关键改动 |
|------|---------|---------|
| **TS-linear** | 连续价格 + 线性需求 | LP → QP，决策变量是连续价格而非分布 |
| **TS-contextual** | 情境化特征信息 | LP 中加入情境分布期望和分情境决策变量 |
| **TS-BwK** | 通用带资源约束的 Bandit | 类比 TS-fixed，但资源用完即停止 |

### 2.6 理论结果：Bayesian Regret 界

论文使用 **Bayesian Regret** 作为性能度量：

$$\text{BayesRegret}(T) = \mathbb{E}_\theta[\mathbb{E}[Rev^*(T)|\theta] - \mathbb{E}[Rev(T)|\theta]]$$

其中 $Rev^*(T)$ 是已知需求参数时的最优期望收入。

| 算法 | Bayesian Regret 界 | 与下界的差距 |
|------|--------------------|-------------|
| TS-fixed | $O(\sqrt{TK\log K})$ | **匹配** $\Omega(\sqrt{T})$ (仅差 $\sqrt{\log K}$) |
| TS-update | $O(\sqrt{TK\log K}) + p_{\max}M$ | 同上，多 $M$ 常数项 |
| TS-linear | $O(N^2 \log T \sqrt{T})$ | 匹配线性 Bandit 下界 |
| TS-contextual | $O(\sqrt{|\mathcal{X}|TK\log K})$ | 依赖特征空间大小 |
| TS-BwK | $O(\sqrt{KT\log K}\log T)$ | 多一个 $\log T$ |

**证明的核心思路（3 步）：**

1. **假设无限库存**：计算期望收入 $\mathbb{E}[\sum_t \sum_k r_k x_k(t)]$
2. **减去缺货损失**：缺货损失 ≤ $\sum_j p^j_{\max} \mathbb{E}[(\sum_i \sum_t a_{ij} D_i(t) - I_j)^+]$
3. **分别 bound 两部分**：
   - 收入项：利用 UCB 函数 $U_k(t)$ 分解为两个可 control 的部分
   - 库存项：将超支分解为随机波动 + 采样偏差，同样用 UCB/LCB 技术

**关键技术亮点：** 论文改造了 Russo & Van Roy (2014) 的分析工具。原工具处理的是**线性可加**的奖励，而缺货损失是**非线性**的（因为库存耗尽时间本身是随机变量）。论文展示了如何用 UCB/LCB 来 bound 非线性损失。

---

## 三、实验设计

### 3.1 实验 1：单产品实验（复现 Figure 1）

**目的：** 验证 TS-fixed 和 TS-update 在基础场景下的有效性，并与基准算法对比。

**参数设置：**
```
N = 1（单产品）
M = 1（单资源）
K = 4（四个价格选项）
价格集合 = {$29.90, $34.90, $39.90, $44.90}
均值需求 d = [0.8, 0.6, 0.3, 0.1]
需求分布：伯努利分布（每个顾客要么买要么不买）
先验分布：Beta(1, 1) 即 Uniform[0, 1]
库存：I = αT, α ∈ {0.25, 0.5}
T ∈ {500, 1000, 2000, 5000, 10000}
仿真次数：500 次
```

**对比算法：**
| 算法 | 简称 | 说明 |
|------|------|------|
| TS-fixed | Ours | Algorithm 1，Beta 先验 |
| TS-update | Ours | Algorithm 2，Beta 先验 |
| Besbes & Zeevi (2012) | BZ | 探索-利用分离，$\tau = T^{2/3}$ 分界 |
| Badanidiyuru et al. (2013) | PD-BwK | UCB-based + Primal-Dual |
| 经典 Thompson Sampling | TS | 无库存约束（对比基准，应不收敛） |

**度量指标：**
- 相对最优收入百分比 = 算法收入 / $OPT(d) \times T$（LP 上界） × 100%
- Bayesian Regret 随 T 的变化曲线（T 从 200 开始画，避免早期噪声过大）

**注：** $OPT(d)$ 需在实验开始前用**真实需求参数**计算一次 LP(d) 得到，所有仿真复用同一基准值。

**预期结果：**
- TS-update > TS-fixed > PD-BwK > BZ >> TS（无约束）
- TS 不收敛，因为最优策略需要混合价格而非单一价格
- $I = 0.25T$ 时，算法间差距更大（库存更稀缺）

### 3.2 实验 2：多产品实验（复现 Figure 2）

**目的：** 验证算法在网络结构下的表现，测试三种需求函数。

**参数设置：**
```
N = 2（两种产品）
M = 3（三种资源）
资源消耗矩阵 A：
  产品1: a = (1, 3, 0)
  产品2: a = (1, 1, 5)

K = 5 个价格向量：
  p ∈ {(1, 1.5), (1, 2), (2, 3), (4, 4), (4, 6.5)}
```

**三种需求函数：**

| 类型 | 公式 | 说明 |
|------|------|------|
| 线性 | $\mu(p) = (8 - 1.5p_1, 9 - 3p_2)$ | 最简单，线性需求 |
| 指数 | $\mu(p) = (5e^{-0.5p_1}, 9e^{-p_2})$ | 非线性弹性 |
| Logit | $\mu_1 = \frac{10e^{-p_1}}{1+e^{-p_1}+e^{-p_2}}, \mu_2 = \frac{10e^{-p_2}}{1+e^{-p_1}+e^{-p_2}}$ | 离散选择模型 |

**对比算法：** TS-fixed, TS-update, BZ（PD-BwK 不适用于泊松到达）

**特殊注意事项：**
- 多产品需要 Gamma 先验（需求为泊松分布，Gamma 是共轭先验）
- 后验更新：每个 $(i,k)$ 组合有独立的 Gamma 后验，$Gamma(W_{ik}(t-1) + 1, N_k(t-1) + 1)$
  其中 $W_{ik}$ 是产品 $i$ 在价格 $k$ 下的累计总需求，$N_k$ 是价格 $k$ 被选择的次数

### 3.3 实验 3（加分项）：自己设计的扩展场景

**选项 A：时变需求环境**
- 需求参数 $\theta$ 在一个销售季节中改变一次（如旺季-淡季切换）
- 测试算法能否快速适应

**选项 B：更大规模的产品网络**
- $N=5, M=5, K=10$ 的中等规模问题
- 测试算法在更大维度下的 Scalability

**选项 C：展示 TS-update 未必优于 TS-fixed**
- 固定 $T$（如 $T=1000$），扫描 $\alpha$ 从 0.1 到 0.9
- 绘制 TS-update 与 TS-fixed 性能差（$\Delta$% of optimal revenue）随 $\alpha$ 的变化曲线
- 预期：低库存时 $\Delta > 0$（update 占优），高库存时 $\Delta \to 0$ 乃至 $\Delta < 0$ 可能出现
- 结合 Cooper (2002) 的 sequential inconsistency 结论说明：re-solving 不保证改善性能（对应论文 Remark 3）

**选项 D：ε-greedy 基准**
- 增加一个简单基准算法：以 $1-\varepsilon$ 概率选择当前最优 LP 解，以 $\varepsilon$ 概率随机探索

**选项 E：情境化信息（TS-contextual）验证**
- 设置两类顾客（如会员/非会员），需求弹性不同
- 取 $|\mathcal{X}| = 2$，实现成本低，直接对应论文 Section 4.2
- 能展示论文扩展部分的价值，适合课程要求的"扩展"环节

### 3.4 输出图表清单

| 图表 | 对应论文 | 内容 |
|------|---------|------|
| 图 1 | Figure 1 | 单产品：% of optimal revenue vs. T, 不同 α |
| 图 2 | Figure 2 | 多产品：% of optimal revenue vs. T, 三种需求函数 |
| 图 3 | 新增 | Regret vs. T（对数坐标），展示 $O(\sqrt{T})$ 量级 |
| 图 4 | 新增 | 热力图：横轴 = $\alpha$（库存比例 0.1~0.9），纵轴 = $T$，颜色 = 最优收入百分比（单一算法）；或 TS-update 相对 BZ 的性能提升幅度（对比） |
| 图 5 | 新增 | $\Delta$ 性能 vs. $\alpha$：固定 $T$ 下 TS-update 与 TS-fixed 之差随库存比例的变化曲线 |
| 表 1 | 新增 | T=10000 时各算法的最终最优收入百分比 |

---

## 四、代码设计与结构

### 4.1 总体架构

```
project/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖（numpy, scipy, matplotlib, cvxpy）
├── config.py                    # 全局配置与参数
├── main.py                      # 入口：运行实验并生成图表
│
├── models/
│   ├── __init__.py
│   ├── demand.py                # 需求分布模型
│   ├── inventory.py             # 库存管理
│   ├── revenue_network.py       # 网络收益管理环境
│   └── posterior.py             # 后验分布更新
│
├── algorithms/
│   ├── __init__.py
│   ├── base.py                  # 算法基类
│   ├── ts_fixed.py              # Algorithm 1: TS-fixed
│   ├── ts_update.py             # Algorithm 2: TS-update
│   ├── ts_linear.py             # Algorithm 3: TS-linear（扩展）
│   ├── ts_contextual.py         # Algorithm 4: TS-contextual（扩展）
│   ├── ts_bwk.py                # Algorithm 5: TS-BwK（扩展）
│   ├── bz.py                    # Besbes & Zeevi (2012) 基准
│   ├── pd_bwk.py                # Badanidiyuru et al. (2013) 基准
│   └── ts_unconstrained.py      # 经典 TS 基准
│
├── experiments/
│   ├── __init__.py
│   ├── config_single.py         # 单产品实验参数
│   ├── config_multi.py          # 多产品实验参数
│   └── runner.py                # 实验运行器
│
├── analysis/
│   ├── __init__.py
│   ├── baseline.py               # LP(d) 最优基准计算（用真实需求参数，预计算一次，所有仿真复用）
│   ├── regret.py                # Regret 计算与分析
│   ├── visualizer.py            # 画图工具
│   └── metrics.py               # 性能指标
│
└── utils/
    ├── __init__.py
    ├── lp_solver.py             # LP 求解封装
    └── statistics.py            # 统计工具（置信区间等）
```

### 4.2 核心模块设计

#### 4.2.1 `models/revenue_network.py` — 环境模拟

```python
class RevenueNetwork:
    """
    网络收益管理环境
    负责：库存管理、需求生成、收入计算
    """
    def __init__(self, N, M, K, A, prices, I, T, demand_model, theta):
        self.N = N          # 产品数
        self.M = M          # 资源数
        self.K = K          # 价格向量数
        self.A = A          # 资源消耗矩阵 [N x M]
        self.prices = prices # 价格向量 [K x N]
        self.I = I          # 初始库存 [M]
        self.T = T          # 销售季长度
        self.demand_model = demand_model
        self.theta = theta  # 真实需求参数

    def reset(self):
        """重置库存到初始状态"""

    def step(self, price_idx):
        """
        给定所选价格索引，返回需求、收入、是否缺货
        处理两种情况：
        (a) 库存充足：所有需求被满足
        (b) 库存不足：部分满足，部分丢失
        """

    def get_inventory_levels(self):
        """返回当前库存水平"""
```

#### 4.2.2 `algorithms/base.py` — 算法基类

```python
class DynamicPricingAlgorithm(ABC):
    """
    所有定价算法的抽象基类
    """
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def initialize(self, env):
        """算法初始化"""

    @abstractmethod
    def choose_price(self, t, env):
        """
        根据历史和当前状态选择价格
        返回：price_idx
        """

    @abstractmethod
    def update(self, t, price_idx, demand):
        """
        观察需求后更新内部状态/后验
        """
```

#### 4.2.3 `algorithms/ts_fixed.py` — TS-fixed 核心

```python
class TSFixed(DynamicPricingAlgorithm):
    def __init__(self, N, M, K, prices, A, I, T, prior_type='beta'):
        """
        N: 产品数, M: 资源数, K: 价格向量数
        prices: [K x N] 价格矩阵
        A: [N x M] 资源消耗矩阵
        I: [M] 初始库存
        T: 销售季长度
        """
        self.c = I / T  # 固定库存率 [M]
        # 后验参数（例如 Beta 分布的 α, β）
        self.posterior_params = self._init_posterior(prior_type)

    def choose_price(self, t, env):
        d_sample = self._sample_demand()    # 步骤 1
        x_opt = self._solve_lp(d_sample)    # 步骤 2
        price_idx = np.random.choice(K, p=x_opt)  # 步骤 3
        return price_idx

    def update(self, t, price_idx, demand):
        self._update_posterior(price_idx, demand)  # 步骤 4

    def _solve_lp(self, d_sample):
        """
        求解 LP(d(t)):
        max Σ_k (Σ_i p_ik * d_ik) * x_k
        s.t. Σ_k (Σ_i a_ij * d_ik) * x_k ≤ c_j, ∀j
             Σ_k x_k ≤ 1
             x_k ≥ 0

        使用 scipy.optimize.linprog 配合 method='highs'（scipy 1.7+ 默认）或 cvxpy
        """
```

#### 4.2.4 `algorithms/bz.py` — BZ 基准算法

```python
class BZAlgorithm(DynamicPricingAlgorithm):
    """
    Besbes & Zeevi (2012)
    探索-利用分离策略
    """
    def __init__(self, ...):
        self.tau = int(T ** (2/3))  # 探索/利用分界点

    def choose_price(self, t, env):
        if t <= self.tau:
            # 探索阶段：按模 K 循环均等试验每个价格 floor(tau/K) 或 ceil(tau/K) 次
            return self._exploration_price(t, self.K)
        else:
            # 利用阶段：基于探索阶段的数据求解 LP
            return self._exploitation_price()

    def _exploration_price(self, t, K):
        """按模 K 循环确保各价格均等探索"""
        return (t - 1) % K
```

#### 4.2.5 `utils/lp_solver.py` — LP 求解封装

```python
def solve_pricing_lp(prices, mean_demand, A_consumption, c):
    """
    求解 TS-fixed 中的 LP(d)

    参数:
        prices: [K, N] 价格矩阵
        mean_demand: [N, K] 均值需求矩阵
        A_consumption: [N, M] 资源消耗矩阵
        c: [M] 库存率

    返回:
        x_opt: [K] 最优价格选择概率
        opt_value: 最优目标值
    """
    # 用 scipy.optimize.linprog 实现
    # 注意 scipy 的标准形式是 min c^T x, 需要将目标取负
```

#### 4.2.6 `models/posterior.py` — 后验更新

```python
def update_beta_posterior(alpha, beta, price_idx, demand):
    """
    Bernoulli 需求 + Beta 先验
    后验: Beta(α + 购买数, β + 未购买数)
    调用前须先更新 N_k 计数器
    """

def update_gamma_posterior(W_ik, N_k):
    """
    Poisson 需求 + Gamma 先验
    调用前须先完成：
      W_ik += demand_i    # 更新产品 i 在价格 k 下的累计需求
      N_k += 1            # 更新价格 k 被选次数
    后验: Gamma(W_ik + 1, N_k + 1)   # +1 来自初始 Gamma(1,1) 先验
    """
```

### 4.3 实验运行流程

```python
def run_single_simulation(config, algorithm_class, env_seed, algo_seed):
    """
    一次完整的仿真实验流程
    env_seed: 控制需求实现的随机性
    algo_seed: 控制算法内部采样（Thompson Sampling 等）的随机性
    """
    # 1. 初始化环境（使用 env_seed 控制需求序列）
    env = RevenueNetwork(config, seed=env_seed)
    env.reset()

    # 2. 初始化算法（使用 algo_seed 控制内部随机性）
    algo = algorithm_class(config, seed=algo_seed)
    algo.initialize(env)

    # 3. 运行 T 个周期
    revenues = []
    for t in range(1, config.T + 1):
        price_idx = algo.choose_price(t, env)
        demand, revenue, _ = env.step(price_idx)
        algo.update(t, price_idx, demand)
        revenues.append(revenue)

    # 4. 计算累计收入和 regret
    return np.sum(revenues)

def run_experiment(config, algorithm_classes, n_simulations=500):
    """
    运行多次仿真并汇总统计
    """
    results = {name: [] for name in algorithm_classes}
    for sim in range(n_simulations):
        for algo_idx, (name, algo_cls) in enumerate(algorithm_classes.items()):
            # 环境种子控制需求实现，算法种子控制内部随机性
            # 分离种子确保公平比较且算法间随机性独立
            env_seed = sim
            algo_seed = sim + 10000 * algo_idx
            rev = run_single_simulation(config, algo_cls, env_seed, algo_seed)
            results[name].append(rev)

    # 计算均值、标准差、置信区间
    return aggregate_results(results)
```

### 4.4 关键挑战与解决方案

| 挑战 | 解决方案 |
|------|---------|
| LP 求解可能为空 | 使用"关闭价格" $p_\infty$ 确保总有可行解 |
| 后验分布大规模更新 | 使用共轭先验（Beta-Bernoulli, Gamma-Poisson）保持计算效率 |
| 500 次仿真 × 大 T 很慢 | 实验早期用较少仿真（100 次）验证，最终跑全量 |
| LP 求解器在边缘参数可能不稳定 | 添加小量 $\epsilon$ 确保数值稳定性 |
| "缺货"的精确建模 | 遵循论文的两种情况：部分满足 vs 全部满足 |

---

## 五、评估与交付

### 5.1 核心验证标准

| 验证点 | 通过标准 |
|--------|---------|
| 单产品收敛性 | TS-fixed/update 收敛到 95%+ 最优收入 |
| 多产品收敛性 | T=10000 时达到 99%+ |
| BZ 算法性能 | 与论文一致：在 TS 算法之下 |
| TS 无约束不收敛 | 明显低于有约束算法 |
| $\sqrt{T}$ regret 量级 | 在对数坐标下呈 1/2 斜率 |
| 仿真可重复性 | 固定种子后结果完全一致 |

### 5.2 报告结构建议

```
1. 引言
   - 问题背景：在线零售商动态定价
   - 探索-利用权衡与库存约束
   - 论文贡献与本文目标

2. 问题建模
   - 网络收益管理模型形式化
   - 符号表
   - 与 MAB 问题的关系

3. 算法设计
   - 3.1 Thompson Sampling 回顾
   - 3.2 TS-fixed：算法 + 解释
   - 3.3 TS-update：改进点
   - 3.4 扩展算法（简述）

4. 理论分析
   - 4.1 Bayesian Regret 定义
   - 4.2 LP 基准上界
   - 4.3 主要定理与直观解释
   - 4.4 证明要点概述（UCB分解）

5. 数值实验
   - 5.1 单产品实验：设置 + 结果 + 分析
   - 5.2 多产品实验：设置 + 结果 + 分析
   - 5.3 扩展实验（加分项）
   - 5.4 与论文结果对比

6. 结论
   - 6.1 主要发现
   - 6.2 方法优势
   - 6.3 方法局限
   - 6.4 未来扩展方向

参考文献
附录：代码与参数细节
```

### 5.3 时间线建议

| 阶段 | 内容 | 产出 |
|------|------|------|
| 第 1-2 周 | 环境搭建 + 基础模型 | 可运行的 TS-fixed 单产品版本 |
| 第 3-4 周 | 所有算法实现 + LP 调优 | TS-update, BZ, PD-BwK 全部跑通 |
| 第 5-6 周 | 系统实验 + 画图 | Figure 1 & 2 的完整复现 |
| 第 7-8 周 | 报告撰写 + 可视化优化 | 最终报告 + PPT |

---

*本文档基于 Ferreira, Simchi-Levi & Wang (2015) "Online Network Revenue Management using Thompson Sampling" 整理*
