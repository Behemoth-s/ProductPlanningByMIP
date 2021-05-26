# 4. 分类和重构

第3章我们简单介绍了基本的分支定界和分支切割算法，但是实际的计划问题往往是产品数量多、生产关系复杂、问题规模大，通用的算法并不能很好的完成求解。本章，我们介绍一些常见的公认的基础模型的重构方法，来帮助读者如何分辨这些子问题，以及这些重构方法如何在分支定界法和分支切割算法发挥作用。

**目的**

- 介绍实际问题中标准的常见的子问题或者简化问题
- 介绍这些问题的重构方法，以及如何设计分支定界法和分支切割算法来提升重构问题的求解效率。
- 对于复杂问题，说明重构问题的有效性，以及如何从标准子问题出发重构问题，设计算法

第5章我们会介绍如何在软件中对问题进行分类和重构，本章的主要内容侧重于机理的推导。

**目录**

- 4.1 介绍LS-U问题的重构
- 4.2 分解方法在问题重构的应用
- 4.3 标准问题的分类模板
- 4.4 单产品的系统重构方法
- 4.5 综合以上方法，对1.2的MPS问题重构案例

## 4.1 LS-U问题的重构

LS-U问题，单产品单层级无限产能问题，是绝大多数排产问题都存在的简化问题类型，因此作为我们的第一步。本节我们介绍一种单产品的重构方法，被称为 先验重构与切割平面和分离方法（什么鬼名字）。

我们仍然使用1.1的自行车生产问题，简化了最后一个月的库存，将其设为0。
$$
\begin{align}
\text{min}\quad &cost = \sum_{t=1}^{NT}(px_t+qy_t)+\sum_{t=1}^{NT-1}hs_t\\
&x_t,s_t\in \R_+, y_t\in\{0,1\} \quad\text{for all}\quad 1\leq t\leq NT
\end{align}
$$

$$
\begin{align}
dem\_sat:\quad &s_{t-1}+x_t=s_t+d_t\quad \text{for all}\quad 1\leq t\leq NT\\
&s_0=ss\_init, s_{NT}=0
\end{align}
$$

$$
\begin{align}
vub:\quad &x_t\leq y_t\sum_{k=t}^{NT}d_t \quad\text{for all} \quad 1\leq t\leq NT\\
\end{align}
$$

其中$NT=8,p=100,q=5000,h=5,s\_init=200$$,d=[400,400,800,800,1200,1200,1200,1200]$

该问题的复杂度为$O(NT)\times O(NT)$

问题使用不带切割的分支定界法求解的问题概况如下

![image-20210114140426013](assets/image-20210114140426013.png)

LP Val表示初始的线性松弛问题的解，Vars 和 Cons表示线性松弛问题的变量个数和约束个数；CPLP Val表示添加切割平面后的线性松弛问题的解，Time和Cust表示耗时和添加的切割平面数量；OPT Val 表示原问题的整数解，Time和Nodes表示耗时和节点个数。

在该问题中，我们会使用根节点的Lower Bound和节点数量来评价重构的好坏，运行时间太短反而不能作为评价标准。

**结论4.1 ** *我们在约束vub中已经使用了一些收紧约束的方法。假如我们使用一个足够大的Big-M来代替约束vub的右项的$\sum_{k=t}^{NT}d_t$，例如$M=10000$。那么该问题初始节点的线性松弛的LowerBound会下降到703500，总节点数会增加到51*

### 4.1.1 先验扩展重构

LS-U问题可以使用动态优化解决的，因而是一个多项式复杂度的问题（对比MIP是一个指数复杂的问题）。我们在3.4提过，优化问题的复杂度与对应的分离问题相当，因此，我们希望可以找到一个多项式复杂度的LS-U问题的分离算法，来收紧问题的约束。我们首先介绍一个著名的LS-U问题的扩展重构方法。

**多商品扩展重构**

LS-U的经典收紧约束方法是将当期产量$x_t$的网络流分解成后续每个周期的产量供应。$x_t$周期内生产的产品会被用在满足$t,\cdots ,NT$后需周期的需求，我们可以假设$x_t$产品在$t$周期内发货和$t+1$周期内发货的是不同产品，把它拆分开来。

例如，将$x_{t},1\leq t\leq NT$分解成$x_{i,t},1\leq i \leq NT,1\leq t\leq NT$。下标$i$表示在$i$周期生产，下标$t$表示货物类型$t$（在$t$周期发货的货物）。显然有$i>t$时$x_{i,t}=0$。

对应的，会有$s_{i,t}$，表示$i$周期末，$t$型货物库存量。显然当$i>t$时，$s_{i, t}=0$。考虑到我们不会希望最后会有剩余库存，可以认为当$i=t$时，$s_{i,t}=0$。

产品需求$d_{i,t}$表示周期$i$需求的货物$t$的数量。对于货物类型$t$，它只会在周期$t$发货，所以$d_{t,t}=d_{t}, 1\leq t\leq NT$，其余皆为0。

变量$y_i$则无需修改，仍然可以表示当期是否生产。

更进一步的，我们发现对于$i>t$的索引部分无需定义。

所以该问题的索引为$\{i,t\}, 1\leq i \leq NT, t\geq i$，有变量为$x_{i,t},s_{i,t}, y_i$和参数$d_{i,t}$

目标，这里我们简化了库存费用，因为最后一个月反正都应该是0库存
$$
\text{min}\quad cost=\sum_i\sum_{t\geq i}(px_{i,t}+hs_{i,t})+\sum_iqy_i
$$


对于产量和库存的平衡约束$dem\_sat$
$$
s_{i-1,t}+x_{i,t}=d_{i,t}+s_{i,t}
$$
特别的，对于初始库存，因为小于1月的需求，所以我们认为初始库存全部在1月发货
$$
s_{0, 1}= s\_init
$$
后续的初始库存全部为0
$$
s_{0, t}=0,\quad 2\leq t \leq NT
$$
我们不希望有剩余库存（其实可以不用显示约束，因为在优化目标上劣势）
$$
s_{t, t}=0， 1\leq t \leq NT
$$
产量上限约束$vub$，产量应当小于需求的上限，将产量分解之后，我们变量的上限不再是$d$累加值，而是$d_{t}$，这极大的减小了变量范围。
$$
x_{i, t} = (d_{t}-s_{0,t})y_i
$$

**重构结果**

原问题结果

```
Objective value:                736000
Model:							14*21
Initialization LR value:        712189
Enumerated nodes:               24
Total iterations:               35
Time (CPU seconds):             0.13
Time (Wallclock seconds):       0.13

Total time (CPU seconds):       0.16   (Wallclock seconds):       0.16
```

重构问题结果

```
Objective value:                736000
Model:							33*40
Initialization LR value:        736000
Enumerated nodes:               0
Total iterations:               0
Time (CPU seconds):             0.04
Time (Wallclock seconds):       0.04

Total time (CPU seconds):       0.12   (Wallclock seconds):       0.12
```

可以发现，原问题的初始线性解为712189，节点数为24，问题规模为$O(NT)\times O(NT)$；重构后的问题初始线性解为736000，节点为0，问题规模为$O(NT^2)\times O(NT^2)$。重构后的问题的初始节点线性解即为最优解。在该问题中，我们通过增加变量和约束的方式，讲$O(2^n)$复杂度的整数规划，简化为一个$O(n^2)$的线性规划问题。

**定理 4.1** *本节的LS-U重构方式，直接找到了整数最优解，因此该重构即是LS-U问题的凸包的重构，又被称为LS-U问题的完全线性描述。*

 原问题代码

```julia
using JuMP
using Cbc
model = Model(Cbc.Optimizer)
set_optimizer_attribute(model, "cuts","off")

month_num = 8
month = 1:month_num
# paramerters
q = [5000] # product fix cost
p = [100] # cost per product
d = [400, 400, 800, 800, 1200, 1200, 1200, 1200] # demand of product by month
s_init = [200] # initial stock
h = [5] # stock cost per product month

@variable(model, x[j in month] >= 0)
@variable(model, s[j in month] >= 0)
@variable(model, y[j in month], Bin)

# 库存平衡
@constraint(model ,dem_sat1, s_init[1] + x[1] == d[1] + s[1])
@constraint(model, dem_sat[j in month[2:end]], s[j - 1] + x[j] == d[j] + s[j] )
# 最大产量约束
@constraint(model, vub[j in month], x[j] <= sum(d[k]  for k in j:month_num) * y[j])
# 目标函数
@expression(model ,cost, sum(p[1] * x[j] + q[1] * y[j] for j in month) + sum(h[1] * s[j] for j in month[1:end - 1]))
@objective(model, Min, cost)
optimize!(model)
# 显示结果
cost_val = value.(cost)
x_val = value.(x)
s_val = value.(s)
@show cost_val;
@show x_val;
@show s_val;
```

重构问题代码

```julia
using LinearAlgebra
using JuMP
using Cbc

model = Model(Cbc.Optimizer)
set_optimizer_attribute(model, "cuts","off")

month_num = 8
month = 1:month_num
# paramerters
q = [5000] # product fix cost
p = [100] # cost per product
d = diagm([400, 400, 800, 800, 1200, 1200, 1200, 1200]) # demand of product by month
s_init = [200,0,0,0, 0,0,0,0] # initial stock
h = [5] # stock cost per product month

@variable(model, x[i in month, t in month] >= 0)
@variable(model, s[i in month, t in month] >= 0)
@variable(model, y[i in month], Bin)

# 库存平衡
@constraint(model ,dem_sat1[t in month], s_init[t] + x[1,t] == d[1, t] + s[1, t])
@constraint(model, dem_sat[i in month[2:end], t in month], s[i - 1, t] + x[i, t] == d[i, t] + s[i, t] )

# 最大产量约束
@constraint(model, vub1[i in month], x[i, 1] <= (d[1,1] - s_init[1]) * y[i])
@constraint(model, vub[i in month, t in month[2:end]], x[i, t] <= d[t,t] * y[i])
# 目标函数
@expression(model ,cost, sum(p[1] * x[i,t] + h[1] * s[i, t] for i in month, t in month[i:end]) + sum(q[1] * y[i] for i in month))
@objective(model, Min, cost)
for t in month
    fix(s[t,t], 0;force=true)
end
optimize!(model)

# 显示结果
cost_val = value.(cost)
x_val = value.(x)
s_val = value.(s)
@show cost_val;
@show x_val;
@show s_val;
```

### 4.1.2 使用切割平面

先验扩展重构问题可以将LS-U问题重构为$O(NT^2)\times O(NT^2)$的扩展重构问题，但是对于物品众多时间维度大的LS-U问题，扩展重构的规模可能也是过大的。这里，介绍LS-U问题的一种有效不等式的生成方法以及对应的分离和切割平面算法。

**有效不等式**

![image-20210120102636135](https://gitee.com/behe-moth/picgo_img/raw/master/pic/20210120102638.png)

我们求解原问题的LR问题，得到各个$y$的非整数值和$x,s$表示如图4.2。以一个非整数解$y_2=0.059$为例，在这里$y_2$受约束$x_2\leq y_2\sum_{t=2}^8d_t$限制，取到了其最小值。定义$d_{\alpha,\beta}=\sum_{t=\alpha}^\beta d_t$，该约束可以表示为$y_2\geq x_2 /d_{2,8}$。

我们观察到对于$y_8$，$y_8\geq x_8/d_8=1$，刚好可以取到整数。如果$x_2$也是最后一个周期的话，那么似乎对于当前的$x_2=400$来说，我们也可以得到$y_2\geq x_2/d_2=1$。我们引入$s_2$，即可将$x_2$视为一个完整结束的周期，得到

$$
x_2\leq y_2d_2+s_2
$$
我们得到了一条有效不等式，当$y_2=0$时,约束$vub$会限制$x_2=0$；当$y_2=1$时，会约束上限$x_2\leq d_2+s_2$。

更进一步，我们可以将该约束扩展到一个间隔范围$l\geq t$
$$
x_t \leq d_{t, l}y_t + s_l \quad 1\leq t\leq l \leq NT
$$
**完全线性描述**

我们使用$X^{LS-U}$表示LS-U的原始空间，其凸包为$\text{conv}(X^{LS-U})$。式13还不足以完全描述LS-U的凸包，其具体如下，详细推导可以见第7章。该式被称为$(l,S)$不等约束
$$
\sum_{i\in S}x_i \leq \sum_{i\in S}d_{i,l}y_i+s_l \quad 1\leq l\leq NT, S\sube\{1,\cdots, l\}
$$
例如，当$l=4, S\sube\{1,2,3,4\}=\{2,3\}$时，有$x_2+x_3 \leq d_{2,4}y_2+d_{3,4}y_4+s_4$

**定理 4.2**  假设$d_t\geq 0, s\_init = 0$，凸包$\text{conv}(X^{LS-U})$的完全描述如下：
$$
s_{t-1}+x_t = d_t +s_t
$$

$$
s_0=0, s_{NT}=0
$$

$$
x_t \leq d_{t, NT}y_t
$$

$$
\sum_{i\in S}x_i \leq \sum_{i\in S}d_{i,l}y_i+s_l \quad 1\leq l\leq NT, S\sube\{1,\cdots, l\}
$$

$$
x_t,y_t,s_t \in \R_+, y_t\leq 1
$$

简化情况是，当$d_1>0$时，有$y_1=1$。

**分离算法**

我们通过有效不等式获得了凸包的完全描述，当问题在于，$(l,S)$有效不等式的数量是指数爆炸的，因此我们需要使用分离算法和切割平面法来对有效不等式进行精简，只选择那些受约束的不等式进行重构。

**分离问题** 对于$(x^*,y^*,s^*)\in P^{LS-U}$,找到一条不满足的$(l,S)$不等式，或者证明其满足所有的$(l,S)$不等式。

由于$\sum_{i\in S}x_i - \sum_{i\in S}d_{i,l}y_i \leq s_l$，该问题可以取其子集$S^*$
$$
S^* = \{i \in \{1,\cdots,l\}:\sum_{i\in S}x_i - \sum_{i\in S}d_{i,l}y_i> 0\}
$$
通过枚举$l$我们可以完成$O(NT^2)$的分离算法，从而精简问题。

该问题的模拟，可以通过启用求解器的cuts功能实现，在CBC上的结果如下

```
Cbc0013I At root node, 7 cuts changed objective from 717050.07 to 736000 in 3 passes
Cbc0014I Cut generator 0 (Probing) - 6 row cuts average 2.0 elements, 0 column cuts (0 active)  in 0.001 seconds - new frequency is 1
Cbc0014I Cut generator 1 (Gomory) - 8 row cuts average 5.1 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is 1
Cbc0014I Cut generator 2 (Knapsack) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0014I Cut generator 3 (Clique) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0014I Cut generator 4 (MixedIntegerRounding2) - 6 row cuts average 3.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is 1
Cbc0014I Cut generator 5 (FlowCover) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0014I Cut generator 6 (TwoMirCuts) - 8 row cuts average 3.4 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is 1

Objective value:                736000
Model:							14*21
Initialization LR value:        712189
Enumerated nodes:               0
Total iterations:               8
Time (CPU seconds):             0.21
Time (Wallclock seconds):       0.21

Total time (CPU seconds):       0.24   (Wallclock seconds):       0.24
```

可以看到，经过7次切割，LR的解从717050.07 到736000，而没有遍历其他节点。

### 4.1.3 近似重构

LS-U是一个简单的问题类型，因此我们有明确的重构方式和求解方法。对于复杂的问题，如果其子问题含有LS-U问题，可以使用前述方法将问题简化。如果只是部分的问题，例如LS问题，这时候我们可能会考虑使用近似的凸包来描述问题。
$$
\text{conv}(X^{LS})\sub \overline{conv}(X^{LS})\sub P^{LS}
$$
接下来我们会讨论如何将这些已知的子问题应用到实际的应用案例的重构当中去。

## 4.2 复杂模型的分解

以MPS问题为例
$$
(\text{MIPP}^{item}) \quad W^* =\text{min} \sum_i\sum_t(p_t^ix_t^i+h_t^is_t^i+q_t^iy_t^i)
$$

$$
s_{t-1}^i+x_t^i=d_t^i+s_t^i
$$

$$
x_t^i\leq C_t^iy_t^i, \quad y_t^i\leq 1
$$

$$
\sum_i\alpha_t^{i,k}+\sum_i \beta_t^{i,k}y_t^i\leq L_t^k
$$

$$
x_t^i\in\R_+^1,s_t^i\in\R_+^1,y_t^i\in\Z_+^1
$$

我们可以将问题表示成如下形式
$$
(\text{MIPP}^{item}) \quad W^* =\text{min} \sum_i\sum_t(p_t^ix_t^i+h_t^is_t^i+q_t^iy_t^i)\\
(x^i,s^i,y^i)\in Y^i\\
(x,s,y)\in Z
$$
$Y^i$表示物品$i$的LS问题的解，满足式23和24；$Z$表示满足式24和25的解，表示物品之间的产能耦合关系。

或者进一步分解
$$
(\text{MIPP}^{item}_{time}) \quad W^* =\text{min} \sum_i\sum_t(p_t^ix_t^i+h_t^is_t^i+q_t^iy_t^i)\\
(x^i,s^i,y^i)\in Y^i\\
(x_t,s_t,y_t)\in Z_t
$$
使用$Z_t$表示不同时间周期内的子模型。

其LR问题可以表示如下
$$
LR \quad =\text{min} \sum_i\sum_t(p_t^ix_t^i+h_t^is_t^i+q_t^iy_t^i)\\
(x^i,s^i,y^i)\in P^{Y^i}\\
(x_t,s_t,y_t)\in P^{Z_t}
$$
由于其原问题过于复杂，无法直接求解，一般我们会将其按照物品分解，并进行近似重构，得到
$$
LB^{item} \quad =\text{min} \sum_i\sum_t(p_t^ix_t^i+h_t^is_t^i+q_t^iy_t^i)\\
(x^i,s^i,y^i)\in \overline{\text{conv}}(Y^i)\\
(x_t,s_t,y_t)\in P^{Z_t}
$$
近似重构可以使用约束收紧方式，或者有效不等式配合分离算法得到。

在有些情况下，还可以对$Z_t$进行近似重构
$$
LB^{item}_{time} \quad =\text{min} \sum_i\sum_t(p_t^ix_t^i+h_t^is_t^i+q_t^iy_t^i)\\
(x^i,s^i,y^i)\in \overline{\text{conv}}(Y^i)\\
(x_t,s_t,y_t)\in \overline{\text{conv}}(Z_t)
$$
以上问题的下限有如下关系
$$
LR\leq LB^{item}\leq LB^{item}_{time}\leq W^*
$$
LB越高意味节点数越少，但获得更高的LB也需要更多的计算。通常我们会根据已有的子问题重构模型，在计算耗时和节点数量之间权衡模型的分解。我们可以根据模型的分类来确定采用何种问题分解方法。

## 4.3 模型的分类

多数的计划问题都是多产品多层级多机器的问题，但很少有这类问题的高性能的重构模型。对于多数的问题，优化的任务主要是将原始问题与已有的算法和重构方法结合，来对问题进行分解，使之达到可以求解的程度。

本章我们会使用GW的单层级问题问题，来展示模型的分解过程。在第二和第三部分，我们会介绍对于单产品的重构结论。

### 4.3.1 单产品的分类

首先对之后用到的符号进行说明。通常，我们使用$n$来表示一般问题中的时间周期的数量，对于具体的规划问题使用$NT$表示时间周期数量。同样，用$m$表示一般问题的物品种类，使用$NI$表示具体问题中的物品种类。其他变量和参数定义如下

- $d_t$ 需求量
- $p'_t$ 生成单位费用
- $q'_t$ 生产固定费用
- $h'_t$ 库存单位费用
- $C_t$ 生产上限

我们对单产品问题的分类依据其难度，确切的说是文献中对其重构的研究和结论。问题可以分为三个部分PROB-CAP-VAR。在后续问题中，我们使用$[x,y,z]^1$表示$\{x,y,z\}$的其中一个元素，$\{x,y,z\}^*$表示$\{x,y,z\}$的任意子集，$x,y,z$表示$\{x,y,z\}$的所有元素。

### 4.3.2 PROB介绍

第一个部分，我们介绍四种问类型PROB=[LS, WW, DLSI, DLS]

LS(lot-sizing)：单产品单层级有容量限制的排产问题

WW (Wagner–Whitin) ：在LS问题的基础上，满足$h'_t+p'_t-p'_{t+1}\geq 0$。该问题中，$t$时刻的生产费用和库存费用大于$t+1$时刻的生产费用，意味着越晚生产费用越低，生产排期会尽可能临近交期。一般我们会定义新的库存费用$h_t = h'_t+p'_t-p'_{t+1}$来处理此类问题。

DLSI (Discrete Lot-Sizing with Variable Initial Stock)  ：在该问题中，每个周期要么不生产，要么满负荷生产。

DLS (Discrete Lot-Sizing)  ：没有初始库存的DLSI问题。

### 4.3.3 CAP介绍

有三种类型，CAP=[C, CC, U]，对应有三种PROB变体

- PROB-C（Capacity）：产能限制随时间变化的问题
- PROB-CC（Constant Capacity）:产能限制不变
- PROB-U(Uncapacity)：产能无限

在介绍VAR和其众多扩展部分之前，我们先介绍PROB-C的4种基本重构形式。

### 4.3.4 PROB-C的数学形式

我们仍然使用$x_t,y_t,s_t$作为变量，和参数$d_{t,n}=\sum_{u=t}^nd_u$。

LS-C的问题模型如下
$$
\text{min} \quad \sum_{t=1}^np'_tx_t+\sum_{t=0}^nh'_ts_t+\sum_{t=1}^nq_ty_t
$$

$$
s_{t-1}+x_t=d_t+s_t
$$

$$
x_t\leq C_ty_t
$$

$$
x\in \R_+^n, s\in \R_+^{n+1},y\in \{0,1\}^n
$$

我们使用$X^{LS-C}$来表示该问题的可行域。

WW-C的问题模型如下：
$$
\text{min} \quad \sum_{t=0}^nh_ts_t+\sum_{t=1}^nq_ty_t
$$

$$
s_{k-1}+\sum_{u=k}^tC_uy_u\geq d_{k,t} \quad 1\leq k\leq t \leq n
$$

$$
s\in\R_+^{n+1}, y\in\{0,1\}^n
$$

我们使用$X^{WW-C}$表示其可行域。根据约束dem-sat，我们可以得到目标函数37的推导如下
$$
\begin{aligned}
\sum_{t=1}^np'_tx_t+\sum_{t=0}^nh'_ts_t & =\sum_{t=1}^np'_t(s_t-s_{t-1}+d_t)+\sum_{t=0}^nh'_ts_t\\
&=\sum_{t=0}^n(h'_t+p'_t-p'_{t+1})s_t+\sum_{t=1}^np'_td_t\\
&=\sum_{t=0}^nh_ts_t+\sum_{t=1}^np'_td_t
\end{aligned}
$$
由于$h_t\geq 0$，一旦确定了$y$，那么库存变量就会倾向最小。可以证明，在$y$确定的情况下，最小库存如下
$$
s_{k-1}=\text{max}(0,\text{max}_{t=k,\cdots,n}[d_{k,t},\sum_{u=k}^tC_uy_u]
$$

**Remark 1 **  无论Wagner-Whitin条件是否成立，约束38对于LS问题都是成立的。约束38通常都会提供一组对凸包表述较好的有效不等式。

**Remark 2**   对于多产品问题，尽管部分产品可能满足WW条件，但是由于其耦合的产能限制，仍然需要当作LS问题处理，而不能直接使用WW。

**Remark 3**   换句话说，如果多产品之间只有固定成本的关联关系没有产能耦合关系。如果满足WW条件，就可以适用WW问题。



DLSI-C的模型如下：
$$
\text{min}\quad h_0s_0+\sum_{t=1}^nq'_ty_t
$$

$$
s_0+\sum_{u=1}^tC_uy_u\geq d_{1,t}
$$

$$
s_0\in\R_+^1,y\in \{0,1\}^n
$$



目标值的推导如下
$$
\sum_{t=0}^nh'_ts_t+\sum_{t=1}^np'_tx_t+\sum_{t=1}^nq_ty_t\\
=h'_0s_0+\sum_{t=1}^nh'_t(s_0+\sum_{u=1}^tC_uy_u-d_{1,t})+\sum_{t=1}^np'_tC_ty_t+\sum_{t=1}^nq_ty_t\\
=(h'_0+\sum_{t=1}^nh'_t)s_0-\sum_{t=1}^nh'_td_{1,t}+\sum_{t=1}^n(q_t+(p'_t+\sum_{u=t}^nh'_u)C_t)y_t
$$
定义$h_0=h'_0+\sum_{t=1}^nh'_t$，$q'_t=q_t+(p'_t+\sum_{u=t}^nh'_u)C_t$，省去常数项，即可得到式41。

另外，我们定义问题$DLSI_k-C$，表示时间周              期$k,\cdots,n$的计划问题，可行域为$X^{DLSI_k-C}$。

DSL-C的模型可以令$s_0=0$将变量简化为$y_t$
$$
\text{min}\quad \sum_{t=1}^nq'_ty_t
$$

$$
\sum_{u=1}^nC_uy_u\geq d_{1,t}
$$

$$
y\in\{0,1\}^n
$$

该问题的可行域定义为$X^{DSL-C}$。当DSL-C问题中，当$q'_t\geq q'_{t+1}$，即满足WW条件，该问题类型为DSL(WW)-C。

**结论4.2 **  *PROB-CC和PROB-U问题都是多项式时间内可解的。其中LS-CC存在多项式时间的动态规划算法，另外其中则可以视为特殊案例。所有的PROB-C都是NP-hard问题，不存在通用的多项式解法*

### 4.3.5 VAR介绍

模型的第三部分是VAR，VAR = [B, SC, ST, LB, SL, SS] 。

B[ Backlogging ] 积压指在交期之后才能满足需求，例如因为运力不足导致的库存积压。

SC[Start-up Cost] 启动费用，一般指机器启动或者切换产品时的固定生产消耗，不会占用机器的总产能，略微不同于固定费用Set-up Cost。

ST[Start-up Time] 启动时间，与开机消耗类似，占用机器的产能，通常来说会比开机消耗更为复杂。

LB (Minimum Production Levels)  通常指每一批次的最小产量

SL (Sales and Lost Sales)  某些情况下，需求的满足条件并不是严格满足，例如因产能不足或者运输损耗导致的需求不满足，这种情况下会增加一项额外的需求$u_t$，其成本为$c_t$，相当于产能不足时的罚款成本或者运输损耗的额外费用。

SS(Safety Stock) 安全库存

### 4.3.6 PROB-CAP-VAR问题的模型

**积压问题 B**

定义$r_t$表示积压量，$b'_t$表示积压费用

LS-C-B问题的模型如下，假设$r_0=0$
$$
\text{min} \quad \sum_{t=0}^nh'_ts_t+\sum_{t=1}^nb'_tr_t+\sum_{t=1}^np'_tx_t+\sum_{t=1}^nq_ty_t
$$

$$
s_{t-1}-r_{t-1}+x_t=d_t+s_t-r_t
$$

$$
x_t\leq C_ty_t
$$

$$
x,r\in \R_+^n, s\in \R_+^{n+1},y\in\{0,1\}^n
$$

WW-C-B问题在LS-C-B的问题基础上，需要满足WW条件$h_t=p'_t+h'_t-p'_{t+1}\geq 0$和$b_t=p'_{t+1}+b'_t-p'_t\geq 0$。WW-C-B问题是WW-C问题的扩展形式，因此同时满足WW-C问题的约束，该约束只包含$s,r,y$变量
$$
\text{min}\quad \sum_{t=0}^nh_ts_t+\sum_{t=1}^nb_tr_t+\sum_{t=1}^nq_ty_t
$$

$$
s_{k-1}+r_l+\sum_{u=k}^lC_uy_u\geq d_{k,l}, \quad 1\leq k\leq l\leq n
$$

$$
s\in \R_+^{n+1}, r\in \R_+^n, y\in \{0,1\}^n
$$

当因为$s_{k-1}= \text{max}_{l\geq k}[d_{k,l}-r_l-\sum_{u=k}^lC_uy_u]^+$（或者$r_l=\text{max}_{k\leq l}[d_{k,l}-\sum_{u=k}^lC_uy_u-s_{k-1}]^+$）时，WW-C-B问题的解，又被称为最小库存解。

DLSI-C-B问题扩展自DLSI-C问题，去除了$s$变量，变量空间$(s_0,r,y)$
$$
\text{min}\quad s_0h_0+\sum_{t=1}^nb'_tr_t+\sum_{t=1}^nq_ty_t
$$

$$
s_0+r_t+\sum_{u=1}^tC_uy_u\geq d_{1,t}
$$

$$
s_0\in \R_+^1,r\in \R_+^n,y\in[0,1]^n
$$

DSL-C-B问题就是当$s_0=0$时的情况。

**启动费用 SC**

在SC问题中，我们引入一个变量$z_t$表示是否存在启动费用，$g_t$表示启动费用。

LS-C-SC的问题形式如下：
$$
\text{min} \sum_{t=1}^np'_tx_t+\sum_{t=0}^nh'_ts_t+\sum_{t=1}^nq_ty_t+\sum_{t=1}^ng_ty_t
$$

$$
s_{t-1}+x_t=d_t+s_t, 1\leq t\leq n
$$

$$
x_t\leq C_ty_t,1\leq t\leq n
$$

$$
z_t\geq y_t-y_{t-1},1\leq t\leq n
$$

$$
z_t\leq y_t, 1\leq t \leq n
$$

$$
z_t\leq 1-y_{t-1},1\leq t \leq n
$$

$$
x\in \R_+^n,s\in \R_+^{n+1},y,z\in\{0,1\}^n
$$

在这里，我们定义$z_t=1$当且仅当$y_t=1,y_{t-1}=0$，或者等价于$z_t=y_t(1-y_{t-1})$。由于约束并不是线性的，因此将其重构为式61-63形式。

对于$[WW,DLSI,DLS]^1-C-SC$的问题，同样在基础问题上引入约束61-63即可。

**启动时间 ST**

基本LS-C-ST问题同样需要新增变量$z_t$表示启动时间是否存在和$ST_t$表示启动时间。该问题的约束基本与LS-C-SC基本相同
$$
\text{min} \sum_{t=1}^np'_tx_t+\sum_{t=0}^nh'_ts_t+\sum_{t=1}^nq_ty_t
$$

$$
s_{t-1}+x_t=d_t+s_t, 1\leq t\leq n
$$

$$
x_t\leq C_ty_t-ST_tz_t,1\leq t\leq n
$$

$$
z_t\geq y_t-y_{t-1},1\leq t\leq n
$$

$$
z_t\leq y_t, 1\leq t \leq n
$$

$$
z_t\leq 1-y_{t-1},1\leq t \leq n
$$

$$
x\in \R_+^n,s\in \R_+^{n+1},y,z\in\{0,1\}^n
$$

**最小生产水平 LB**

LS-C-LB的模型在LS-C的基础上扩展得到
$$
\text{min} \sum_{t=1}^np'_tx_t+\sum_{t=0}^nh'_ts_t+\sum_{t=1}^nq_ty_t
$$

$$
s_{t-1}+x_t=d_t+s_t
$$

$$
x_t\leq C_ty_t
$$

$$
x_t\geq LB_ty_t
$$

$$
x\in\R_+^n,s\in\R_+^{n+1},y\in\{0,1\}^n
$$

对于$[WW,DLSI,DLS]^1-C-ST$的问题，同样在基础问题上引入约束75即可。

**销售 SL**

考虑销售的LS-C-SL问题，新增变量$v_t$表示销售订单数量，$c_t$表示订单的收入。
$$
\text{max} \sum_{t=1}^n(c_tv_t-p_tx_t)-\sum_{t=0}^nh'_ts_t-\sum_{t=1}^nq_ty_t
$$

$$
s_{t-1}+x_t=d_t+v_t+s_t
$$

$$
x_t\leq C_ty_t
$$

$$
v_t\leq u_t
$$

$$
x,v\in\R_+^n,s\in\R_+^{n+1},y\in\{0,1\}^n
$$

在这里使用收入-成本，最大化利润作为目标。

**安全库存 SS**

考虑安全库存时，增加对于库存约束即可。
$$
s_t\geq SS_t
$$

### 4.3.7 PROB-CAP-VAR问题的分类

我们已经介绍完了所有的问题类型，可以统一表示为
$$
[LS,WW,DLSI,DLS]^1-[C,CC,U]^1-[B,SC,ST,ST(C),LB,LB(C),SL,SS]^*
$$
PROB和CAP类型可以包含各一种，VAR问题可以包括任意数量，三者组合变成了问题类型。

**结论 4.3** 在LB(C)和ST(C)的前提下，几乎所有的$PROB-[CC,U]^1-VAR$问题都是多项式时间内可以求解的。

具体的单产品的问题类型和其终止复杂度判断的数学证明已经超过了本书的范畴，我们会在第二部分和第三部分介绍一些更具体的数学模型描述。

## 4.4 重构结果

本章会介绍以下的单产品的模型重构结论：

- 基本的$[LS,WW,DLSI,DLS]^1-[U,CC]^1$问题
- 包含积压的$[LS,WW,DLSI,DLS]^1-[U,CC]^1-B$问题
- 包含启动费用的$[LS,WW,DLSI,DLS]^1-[U,CC]^1-SC$问题

我们不会讨论有容量限制的问题重构，因为有约束的计划问题是NP-Hard问题，不存在完全的重构形式。

在结论中，我们会从Formulation,Separation,Optimization三个方面的复杂度来进行评价。

我们使用符号\*表示重构只是部分，\*\*\*表示问题的部分重构也并未发现。

我们会在之后介绍结论中问题的具体重构方程形式，以供读者使用，而不需要任何数学上或者对重构的分析知识。

### 4.4.1 PROB-[U,CC]

![image-20210201161234022](assets/image-20210201161234022.png)

### 4.4.2 PROB-[U,CC]-B

![image-20210201161348058](assets/image-20210201161348058.png)

### 4.4.3 PROB-[U,CC]-SC

![image-20210201161540220](assets/image-20210201161540220.png)

### 4.4.4 重构的步骤

本节简单介绍问题的分类和重构的指导性建议，具体的重构过程我们会在第5章进行详细介绍和示例。

**建议1** 使用4.3节的内容建立初始的数学模型，即按照PROB-CAP-VAR分类进行单产品的建模

**建议2** 对于每一种产品，按照4.4的表格选择最相似的类型进行重构。重构的选择需要在精度和复杂度之间进行一定折中，对于一些模型，我们可以使用其他类型来近似。

- $CC\Rightarrow U$，表格上移，通常可以减少重构问题规模或者cut数量
- $LS\Rightarrow WW$，表格右移，通常可以减少重构问题的规模或者cuts数量
- $WW\Rightarrow \{DLSI_k\}_{k=0,1,\cdots,n-1}$，在表格上右移
- $WW\Rightarrow LS$，在表格上左移

**建议3** 尝试不同的重构方式，并比较计算结果和效率。

根据建议2，我们可以得到相应的可行域关系
$$
X^{prob-cap-SC}\sube X^{prob-cap}
$$

$$
X^{LS-cap-var}\sube X^{WW-cap-prob}
$$

$$
X^{LS-cap-var}\sube X^{WW-cap-var}\sube \bigcap_{k=0}^{n-1}X^{DLSI_k-cap-var}
$$

举例来说，对于WW-CC-B问题，其重构的复杂度为$O(n^3)\times O(n^2)$

我们可以将$CC\Rightarrow U$，移动到表格上方，简化重构复杂度为$O(n^2)\times O(n)$

也可以将$WW\Rightarrow \{DLSI_k\}_{k=0,1,\cdots,n-1}$,移动到表格右侧，单个$DLSI$问题的复杂度为$O(n^2)\times O(n)$但是新问题需要$n-1$个$DLSI$问题，其复杂度仍然是$O(n^3)\times O(n^2)$

## 4.5 重构案例

以1.2节的问题为例
$$
\text{min} \quad inventory=\sum_i\sum_ts_{i,t}
$$

$$
s_{i,t-1}+x_{i,t}=d_{i,t}+s_{i,t}
$$

$$
s_{i,0}=SS_{i,0}, s_{i,t}\geq SS_{i,t}
$$

$$
x_{i,t}\leq M_{i,t}y_{i,t}
$$

$$
\sum_{i\in F^k}\alpha_i^kx_{i,t}+\sum_i\beta_i^ky_{i,t}\leq L^k
$$

