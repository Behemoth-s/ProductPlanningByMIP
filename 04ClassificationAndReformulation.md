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

