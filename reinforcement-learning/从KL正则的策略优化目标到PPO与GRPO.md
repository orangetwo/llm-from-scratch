# 从 KL 正则的策略优化目标到 PPO、GRPO 与 DPO

## 1. 通用模板

给定提示 $`x \sim \mathcal{D}`$，语言模型按当前策略 $`\pi_\theta(\cdot\mid x)`$ 生成回答 $`y`$，任务奖励 $`R(x,y)`$ 给整条回答打分。奖励可以来自人类偏好训练出的奖励模型，也可以来自答案验证器。训练时固定奖励函数和参考策略 $`\pi_{\mathrm{ref}}`$，后者常取监督微调（SFT）后的模型。

**通用目标是最大化期望任务奖励，同时为偏离参考策略付出代价。** 取 $`\beta\gt 0`$，对参数 $`\theta`$ 最大化

```math
\boxed{
J(\theta)=
\mathbb E_{x\sim\mathcal D}\!\left[
  \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[R(x,y)]
  -\beta\,D_{\mathrm{KL}}\!\left(
    \pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)
  \right)
\right]
}
```

这个减号可以从约束推出。记 $`\bar{K}(\theta)`$ 为按提示取平均的 KL，要求 $`\bar{K}(\theta)\le\kappa`$。引入乘子 $`\beta`$ 后，拉格朗日函数为 $`\mathbb{E}[R]-\beta(\bar{K}-\kappa)`$。固定 $`\beta`$ 时，去掉与 $`\theta`$ 无关的常数 $`\beta\kappa`$，就得到上式。$`\beta`$ 表示每单位 KL 扣多少奖励。[Rafailov 等，2023，式 (3)](https://arxiv.org/pdf/2305.18290#page=3)

**从整条回答到单个 token。** 把回答写成 $`y=(y_1,\ldots,y_T)`$，其中 $`T`$ 是回答的 token 数，$`t`$ 从回答的第一个 token 开始计数。

- **状态 $`s_t=(x,y_{\lt t})`$** 是生成第 $`t`$ 个 token **之前**模型看到的上下文，包括提示 $`x`$ 和已经生成的回答前缀 $`y_{\lt t}=(y_1,\ldots,y_{t-1})`$。第一步的回答前缀为空。
- **动作 $`y_t`$** 是这一步实际生成的一个 token。策略 $`\pi_\theta(\cdot\mid s_t)`$ 给出下一个 token 的概率分布。选出 $`y_t`$ 后，把它接到前缀末尾，就得到下一步的上下文 $`s_{t+1}=(x,y_{\le t})`$。

例如，提示 $`x`$ 是“中国的首都是哪里？”。假设分词器把“北京”拆成“北”和“京”两个 token，随后模型生成结束符 EOS。

| 生成步 $`t`$ | 生成前的状态 $`s_t`$ | 本步生成的 token $`y_t`$ |
| --- | --- | --- |
| 1 | 只有提示 $`x`$ | 北 |
| 2 | 提示 $`x`$ 和已生成的“北” | 京 |
| 3 | 提示 $`x`$ 和已生成的“北京” | EOS（结束） |

第 2 步的 $`\pi_\theta(y_2\mid s_2)`$，就是当前策略在看到提示和“北”之后给“京”的概率。计算这一步的对数比时，分母取参考策略在同一上下文中给“京”的概率。

自回归分解给出整条回答的概率，KL 的定义给出对数概率比的期望。

```math
\begin{aligned}
\pi(y\mid x) &= \prod_{t=1}^{T}\pi(y_t\mid s_t),\\
D_{\mathrm{KL}}(p\Vert q) &= \mathbb E_{y\sim p}\!\left[\log\frac{p(y)}{q(y)}\right].
\end{aligned}
```

因此，整条回答的对数概率比可以拆成各步对数概率比之和。代回通用目标，得到

```math
\begin{aligned}
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
  &=\sum_{t=1}^{T}\log
    \frac{\pi_\theta(y_t\mid s_t)}{\pi_{\mathrm{ref}}(y_t\mid s_t)},\\
J(\theta)
  &=\mathbb E_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot\mid x)}
    \left[R(x,y)-\beta\sum_{t=1}^{T}\log
    \frac{\pi_\theta(y_t\mid s_t)}{\pi_{\mathrm{ref}}(y_t\mid s_t)}\right].
\end{aligned}
```

单个采样 token 的对数比可以为负。固定前缀、按当前策略对下一个 token 取期望后，得到该前缀上非负的条件 KL。

## 2. PPO 和 GRPO 共用的裁剪项

每轮采样前，把当前策略保存为 $`\pi_{\mathrm{old}}`$，用它生成一批回答。在这批数据上更新时，旧策略固定。对同一个前缀 $`s`$ 和 token $`a`$，当前策略与旧策略的概率比为

```math
\rho(a,s)=\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}.
```

这个比值来自重要性采样。假设各策略在同一词表上都给每个 token 正概率，对固定前缀就有

```math
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[f(a)]
=\sum_a\pi_{\mathrm{old}}(a\mid s)
   \frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}f(a)
=\mathbb E_{a\sim\pi_{\mathrm{old}}(\cdot\mid s)}[\rho(a,s)f(a)].
```

取 $`f`$ 为采样后固定的优势估计 $`A`$，其正负表示这个动作比基准好还是差。在旧策略采到的前缀上，PPO 的裁剪版本（PPO-Clip）将 $`\rho A`$ 改为以下裁剪项，GRPO 沿用同一写法。[Schulman 等，2017，式 (6)–(7)](https://arxiv.org/pdf/1707.06347#page=3)

```math
C_\epsilon(\rho,A)=
\min\!\left(\rho A,\,
  \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\,A\right).
```

其中 $`\epsilon`$ 是裁剪宽度，$`\mathrm{clip}`$ 将输入截到给定区间内。$`A\gt 0`$ 时，提高 token 概率带来的收益在 $`\rho=1+\epsilon`$ 封顶。$`A\lt 0`$ 时，降低概率带来的收益在 $`\rho=1-\epsilon`$ 封顶。实际概率比仍可越过阈值。沿用旧策略采到的前缀并加入裁剪后，优化的是本轮更新的**代理目标**。

## 3. RLHF-PPO 把参考 KL 计入采样奖励

InstructGPT 把近端策略优化（PPO）用于基于人类反馈的强化学习（RLHF）。采样时 $`\pi_\theta=\pi_{\mathrm{old}}`$，因此可以直接用通用模板中的逐 token 对数比调整奖励。在每个生成位置计入 KL 项，在最后一个位置发放任务奖励。[Ouyang 等，2022，式 (2) 中的 PPO 情形](https://arxiv.org/pdf/2203.02155#page=9)

```math
\tilde r_t=
\mathbf 1_{\{t=T\}}R(x,y)
-\beta\log\frac{\pi_{\mathrm{old}}(y_t\mid s_t)}
                 {\pi_{\mathrm{ref}}(y_t\mid s_t)}.
```

为什么可以在采样时算好奖励？固定 $`x`$，记整条回答的正则化回报为

```math
g_\theta(y)=R(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
```

对它的期望用乘积法则求导，得到

```math
\nabla_\theta\mathbb E_{y\sim\pi_\theta}[g_\theta(y)]
=\mathbb E_{y\sim\pi_\theta}\!
  \left[g_\theta(y)\nabla_\theta\log\pi_\theta(y\mid x)\right]
-\beta\underbrace{\nabla_\theta\sum_y\pi_\theta(y\mid x)}_{0}.
```

最后一项为零，因为概率和为 1。剩下的项用回报给对数概率的梯度加权。在采样点，把各步 $`\tilde{r}_t`$ 相加就得到这个回报的样本值。

PPO 用这批 $`\tilde{r}_t`$ 和价值模型（critic）估计优势 $`\widehat{A}_t`$，然后在本批更新中固定它。令 $`\rho_t=\rho(y_t,s_t)`$，用 $`\widehat{\mathbb{E}}_{\mathrm{batch}}`$ 表示对批内生成 token 取平均，策略模型最大化

```math
J_{\mathrm{PPO}}^{\mathrm{actor}}(\theta)=
\widehat{\mathbb E}_{\mathrm{batch}}
\!\left[C_\epsilon(\rho_t,\widehat A_t)\right].
```

参考 KL 已经通过 $`\tilde{r}_t`$ 进入 $`\widehat{A}_t`$。复用同一批样本时，随 $`\theta`$ 重新计算的是裁剪项里的 $`\rho_t`$。

## 4. GRPO 在裁剪项外显式扣除参考 KL

以 DeepSeekMath 的组相对策略优化（GRPO）为例，采用结果监督，只给完整回答打分。对同一提示 $`x \sim \mathcal{D}`$，从 $`\pi_{\mathrm{old}}(\cdot \mid x)`$ 独立采样 $`G \ge 2`$ 个回答 $`y^{(1)},\ldots,y^{(G)}`$。第 $`i`$ 个回答的长度为 $`T_i`$，任务奖励为 $`R_i=R(x,y^{(i)})`$。将奖励按组标准化，得到 $`\hat{A}_i=(R_i-\bar{R})/\sigma_R`$，其中 $`\bar{R}`$ 和 $`\sigma_R`$ 是组内均值和标准差。同一回答的各 token 共用这个优势，省去单独训练 critic。若全组同分，可约定 $`\hat{A}_i=0`$。[Shao 等，2024，§4.1.1–4.1.2](https://arxiv.org/html/2402.03300v3#S4.SS1)


令 $`s_{i,t}`$ 为第 $`i`$ 个回答的第 $`t`$ 步前缀，$`\rho_{i,t}=\rho(y_{i,t},s_{i,t})`$。用 $`\hat{k}_{i,t}`$ 表示这个位置的参考 KL 估计，GRPO 最大化

```math
J_{\mathrm{GRPO}}(\theta)=
\mathbb E_{x,\,y^{(1)},\ldots,y^{(G)}}\!\left[
  \frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}
  \left(C_\epsilon(\rho_{i,t},\widehat A_i)-\beta\,\widehat k_{i,t}\right)
\right].
```

组内打分只用任务奖励。$`\beta`$ 直接乘在裁剪项外的 KL 估计上。这里 $`1/T_i`$ 让每条回答先对自身的 token 取平均，再对回答取平均。通用模板中的序列 KL 对 token 求和，回答长度不同时，这两种聚合给 token 的权重不同。

原论文在采到的 token 上按下式计算 KL 项。[DeepSeekMath，式 (3)–(4)](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS1)

```math
u_{i,t}=
\frac{\pi_{\mathrm{ref}}(y_{i,t}\mid s_{i,t})}{\pi_\theta(y_{i,t}\mid s_{i,t})},
\qquad
\widehat k_{i,t}=u_{i,t}-1-\log u_{i,t}.
```

固定前缀，记当前分布为 $`p`$、参考分布为 $`q`$。按 $`p`$ 采样时，估计量的期望满足

```math
\begin{aligned}
\mathbb E_p[u] &= \sum_a p(a)\frac{q(a)}{p(a)}=1,\\
\mathbb E_p[\widehat k] &= -\mathbb E_p[\log(q/p)] = D_{\mathrm{KL}}(p\Vert q).
\end{aligned}
```

又因为 $`\log u\le u-1`$，单个样本的 $`\hat{k}`$ 也非负。这一无偏性针对按当前策略采样时的 KL 数值。策略更新后复用旧样本，得到的是代理目标中的近似。

## 5. DPO 从 KL 正则目标得到偏好对损失

直接偏好优化（Direct Preference Optimization，DPO）的一条训练样本是 $`(x,y_w,y_l)`$。给定同一提示 $`x`$，标注者比较两条完整回答，给出一次偏好判断 $`y_w\succ y_l`$。**这里的两个回答都是序列级变量。** [DPO 原论文，第 3 节](https://arxiv.org/pdf/2305.18290#page=3)

- $`y_w`$ 是标注者更偏好、选中的整条回答。下标 $`w`$ 表示 winner。
- $`y_l`$ 是这次比较中未被选中的整条回答。下标 $`l`$ 表示 loser。

把两条回答展开为 token 序列：

```math
\begin{aligned}
y_w &= (y_{w,1},\ldots,y_{w,T_w}),\\
y_l &= (y_{l,1},\ldots,y_{l,T_l}).
\end{aligned}
```

两条回答的长度 $`T_w`$ 和 $`T_l`$ 可以不同。用 $`b\in\{w,l\}`$ 指定其中一条，$`y_{b,t}`$ 表示它的第 $`t`$ 个 token，$`y_{b,\lt t}`$ 是这条回答在该位置之前的前缀。对当前策略或参考策略 $`\pi`$，按自回归分解有

```math
\log\pi(y_b\mid x)
=\sum_{t=1}^{T_b}\log\pi(y_{b,t}\mid x,y_{b,\t t}),
\qquad b\in\{w,l\}.
```

提示 $`x`$ 在每一步都作为条件。标准 DPO 对各条回答的有效 token 对数概率直接求和，不除以回答长度。[作者实现中的序列打分](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L82-L104)

用第 1 节的 $`R(x,y)`$ 表示回答 $`y`$ 在提示 $`x`$ 下的潜在奖励。固定提示 $`x`$，把每条回答的概率 $`p_y`$ 当作可自由选择的变量，记参考概率为 $`q_y=\pi_{\mathrm{ref}}(y\mid x)`$。将通用目标与归一化约束项相加，得到

```math
\sum_y p_y\!\left[R(x,y)-\beta\log\frac{p_y}{q_y}\right]
+\lambda\!\left(\sum_y p_y-1\right).
```

对 $`p_y`$ 求导并令导数为零，得到最优策略。[DPO 原论文，附录 A.1](https://arxiv.org/pdf/2305.18290#page=15)

```math
\begin{aligned}
0 &= R(x,y)-\beta\!\left(\log\frac{p_y}{q_y}+1\right)+\lambda,\\
\pi^\star(y\mid x) &= \frac{q_y e^{R(x,y)/\beta}}{Z(x)},
\qquad Z(x)=\sum_{y'}q_{y'}e^{R(x,y')/\beta}.
\end{aligned}
```

对第二行取对数并移项，就有

```math
R(x,y)=\beta\log\frac{\pi^\star(y\mid x)}{q_y}+\beta\log Z(x).
```

Bradley–Terry 模型假设，标注者选择 $`y_w`$ 的概率为 $`\sigma(R(x,y_w)-R(x,y_l))`$，其中 $`\sigma(z)=1/(1+e^{-z})`$。**两个回答共享同一个提示，所以奖励相减时，$`\beta\log Z(x)`$ 正好抵消。**

用可训练的 $`\pi_\theta`$ 拟合上式中的 $`\pi^\star`$，得到一对回答的隐式奖励差 $`m_\theta`$。标准 DPO 在离线偏好数据集 $`\mathcal{D}_{\mathrm{pref}}`$ 上最小化偏好标签的负对数似然。[Rafailov 等，2023，式 (5)–(7)](https://arxiv.org/pdf/2305.18290#page=4)

```math
\begin{aligned}
m_\theta(x,y_w,y_l)
&=\beta\!\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right],\\[0.6em]
\mathcal L_{\mathrm{DPO}}(\theta)
&=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D_{\mathrm{pref}}}
\!\left[\log\sigma\!\left(m_\theta(x,y_w,y_l)\right)\right].
\end{aligned}
```

训练时，当前策略和冻结的参考策略各给两条完整回答打分，得到四个序列对数概率。先用第 1 节的 KL 系数 $`\beta`$ 把它们组合成一个奖励差 $`m_\theta`$，再对**这一对回答计算一次损失** $`-\log\sigma(m_\theta)`$。梯度经由前面的求和传回当前策略在各回答位置的对数概率。偏好标签直接提供了更新信号，因此标准 DPO 无需单独拟合奖励模型，也用不到 PPO 和 GRPO 的旧策略裁剪比。

## 6. 数值例子

**PPO 与 GRPO 的单个 token。** 假设在同一前缀下，参考策略给这个 token 的概率是 0.10，采样旧策略给 0.20，一次更新后的当前策略给 0.40。取 $`\epsilon=0.2`$，下表各项都按最大化目标的符号书写。

| 计算位置 | 代入数值 |
| --- | --- |
| 两种算法共用的裁剪项 | $`\rho=0.40/0.20=2`$。若 $`A\gt 0`$，该项只计 $`1.2A`$。 |
| PPO 在采样时计入的 token 奖励调整 | $`-\beta\log(0.20/0.10)=-\beta\log 2\approx-0.693\beta`$。 |
| GRPO 在更新时计入的显式 KL 项 | $`u=0.10/0.40=0.25`$，所以 $`-\beta(u-1-\log u)\approx-0.636\beta`$。 |

**DPO 的一对完整回答。** 假设标注者选中了 $`y_w=(\mathrm{A},\mathrm{EOS})`$，另一个回答是 $`y_l=(\mathrm{B},\mathrm{C},\mathrm{EOS})`$，其中 EOS 表示结束。两条回答分别有 2 个和 3 个 token。下表的每个乘数是该位置实际 token 在给定提示和此前 token 后的条件概率。

| 策略 | 选中回答的序列概率 $`\pi(y_w\mid x)`$ | 未选中回答的序列概率 $`\pi(y_l\mid x)`$ | 两列概率之比 |
| --- | --- | --- | --- |
| 参考策略 | 0.4 × 0.5 = 0.20 | 0.4 × 0.5 × 0.25 = 0.05 | 4 |
| 当前策略 | 0.4 × 0.5 = 0.20 | 0.4 × 0.5 × 0.5 = 0.10 | 2 |

取 $`\beta=1`$，用两个序列概率比计算这对回答的 DPO 损失。

```math
m_\theta=\log 2-\log 4=-\log 2,\qquad
\sigma(m_\theta)=\tfrac13,\qquad
-\log\sigma(m_\theta)=\log 3\approx1.099.
```

这里的 1/3 是模型预测标注者会选择 $`y_w`$ 的概率。当前策略仍更常生成 $`y_w`$，但相对参考策略，它把 $`y_w`$ 对 $`y_l`$ 的概率比减半了。增大这个比值就会降低这一对回答的 DPO 损失。
