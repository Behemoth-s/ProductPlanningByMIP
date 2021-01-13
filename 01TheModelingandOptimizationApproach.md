# Product Planning by MIP 笔记

参考书籍

Pochet, Yves, 和Laurence A. Wolsey. *Production Planning by Mixed Integer Programming*. Springer Series in Operations Research and Financial Engineering. New York ; Berlin: Springer, 2006.

# 1 模型和优化方法

本章节通过两个简单案例介绍建模和优化的概念

## 1.1 自行车生产的案例

### 问题描述

某自行车将要投入一条自行车产线和配套库房。自行车需求量一般在全年会随着季节变化，其需求预测量如下。

| Jan  | 400  |
| ---- | ---- |
| Feb  | 400  |
| Mar  | 800  |
| Apr  | 800  |
| May  | 1200 |
| Jun  | 1200 |
| Jul  | 1200 |
| Aug  | 1200 |
| Sep  | 800  |
| Oct  | 800  |
| Nov  | 400  |
| Dec  | 400  |

自行车的生产计划每月执行一次，可以开工或不开工

开工有固定成本set_cost 5000

自行车的生产成本unit_cost 100/辆

未能销售的自行车还需要额外的库存成本，invent_cost 5 /(辆·月)。

由于库存的计算时间是月底，而库存费用计算方式则是按月计算，因此7月底的库存产生的费用成本应当平均分摊到7月15到8月15两个月，换言之7月的的库存费用是六月底和七月底库存的总费用的一般。
$$
\text{inventory cost} = \sum_{t=Jan}^{Aug} \frac{5\times (INV_{t-1}+INV_{t})}{2}
$$
也即初始的库存和最后一个月的库存费用仅需计算半个月。
$$
\text{inventory cost} = 2.5\times INV_0 +\sum_{t=Jan}^{Jul} 5\times INV_{t}＋　2.5\times INV_{Aug}
$$


生产和库存的总费用 = 固定成本 + 变动成本 + 库存成本

现需要按照该需求，计算相应产线容量和库存容量。该容量需要尽可能小并满足销量的需求，同时应当尽可能的减少投产后的生产和库存成本。因此，最小的产线和库存容量，应当是满足最小生产和库存成本时的产量和库存的峰值，而在这里仅需考虑前8个月即可。

### 简单的解

很简单可以给出该案例的两种极端的解

一种可以一次生产出所有的自行车，这样库存成本最大，但固定成本最低，

![image-20201221202538925](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201221202538925.png)

一种则是每月按需生产，这样库存成本为0，但固定成本最高。

![image-20201221203249911](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201221203249911.png)

### 数学模型

显然通过枚举简单解很难覆盖所有的可能解，因此通常使用一个数学模型来描述问题的约束和目标。一般来说，一个数学模型包含集合对象、参数、变量、约束和优化目标五个部分

1. 集合对象

描述变量或参数的维度信息，例如集合t表示不同产品，而参数q(i)则表示不同产品的固定成本，i表示参数q的维度等于i的元素个数。简单来说，就是变量或者参数的下标。

| 名称 | 描述         | 值   |
| ---- | ------------ | ---- |
| t    | 表示不同月份 | 1-8  |
| i    | 表示不同产品 | bike |

2. 参数

   参数指已知的数据。

| 名称   | 索引下标 | 描述                 | 值     |
| ------ | -------- | -------------------- | ------ |
| q      | i        | 生产开工固定成本     | 5000   |
| p      | i        | 单件生产成本         | 100    |
| d      | i, t     | 产品需求量           | 参见表 |
| s_init | i        | 初始库存             | 200    |
| h      | i        | 单件产品单月库存成本 | 5      |

3. 变量

   变量通常 来说就是模型的可变的决策的量，当然很多时候为了改善模型的性质也会引入一些额外的变量，例如本案例中的逻辑变量y的引入，就是为了避免出现通过x大于0来判断是否有固定成本的问题（if else对于求解器来说是无法处理的，除非手动给出梯度和Hessian信息）

| 名称 | 索引下标 | 描述                   | 类型     |
| ---- | -------- | ---------------------- | -------- |
| x    | i, t     | 每种产品每月的产量     | Positive |
| s    | i, t     | 每种产品每月的月底库存 | Positive |
| y    | i, t     | 每种产品每月是否开工   | Binary   |

4. 约束

   变量满足的约束

| 名称    | 索引下标 | 描述                 |
| ------- | -------- | -------------------- |
| dem_sat | i,  t    | 生产消耗和库存平衡   |
| vub     | i, t     | 每月产量小于总需求量 |

- dem_sat
$$
s_{i, t-1} + x_{i, t} = s_{i, t} + d_{i, t} \\
s_{i, 0} = s\_init_i
$$

- vub
$$
x_{i, t} \leq (\sum_{k\ge t}^{8}d_{i, k})\times y_{i, t}
$$

5. 优化目标，由于初始库存是不可变的，所以此处去掉的初始库存的费用

$$
cost = \sum_{i}\sum_{t}(p_{i, t}x_{i,t}+q_{i, t}y_{i, t}) +\sum_i\sum_{t<NT}s_{i, t}h_{i}+\sum_i\frac{s_{i,NT}h_i}{2}
$$


### 模型实现

该问题有唯一最优解，其最优值为736000，解的详情如下表所示。
|    | Jan| Feb| Mar| Apr| May| Jun| Jul| Aug| Total|
|----|----|----|----|----|----|----|----|----|------|
|Demand| 400| 400| 800| 800| 1200| 1200| 1200| 1200|7200|
| Production|  600 |0 |1600| 0| 1200| 1200| 1200| 1200|7000|
|Unit cost  | 60000| 0| 160000| 0| 120000| 120000| 120000| 120000|700000|
|Set-up cost | 5000| 0| 5000| 0| 5000| 5000| 5000| 5000 |30000 |
| End Inventory| 400| 0| 800| 0| 0| 0| 0| 0|  |
|Inv. cost| 2000| 0| 4000| 0| 0| 0 |0| 0| 6000                |

本案例会提供多个OR软件包和语言的实现方式，作为参考和比较以便读者选择合适的工具。后续的大部分案例，仍然会以GAMS实现为主。

#### GAMS实现

```
Set t 'month '        /1*8/;
Set i 'product object' /bike/;
Parameters q(i) 'set-up cost' /bike 5000 /;
Parameters p(i) 'cost per product' /bike 100/;
Table d(i, t) 'demand '
         1       2       3       4       5       6       7       8
bike   400     400     800     800    1200    1200    1200    1200;

Parameters s_init 'initial stock' /bike 200/;
Parameters h(i) 'holding cost per bike month' /bike 5/;

Positive Variable x(i, t);
Positive Variable s(i, t);
Alias (k, t);
Binary Variable y(i, t);
x.lo(i, t)=0;
s.lo(i, t)=0;
Variable cost ;
Equation acost
		dem_sat(i, t)
		dem_sat1(i)
		vub(i, t);

acost .. cost =e=  sum((i, t), p(i) * x(i, t) + q(i) * y(i, t))
                 + sum((i, t)$( ord(t) <=7), h(i)*s(i, t))
                 + sum(i,h(i)/2*s(i,'8'));
dem_sat(i, t)$(ord(t) >= 2) .. s(i, t-1) + x(i, t) =e= d(i, t) + s(i, t);
dem_sat1(i) .. s_init(i) + x(i, '1') =e= d(i, '1') + s(i, '1');
vub(i, t) .. x(i, t) =l= sum(k$(ord(k)>=ord(t)), d(i, k))*y(i, t);

Model ex1 /all/;
Option MIP = CBC;
solve ex1 minimizing cost using mip;
```

结果信息如下

```

MODEL STATISTICS

BLOCKS OF EQUATIONS           4     SINGLE EQUATIONS           17
BLOCKS OF VARIABLES           4     SINGLE VARIABLES           25
NON ZERO ELEMENTS            64     DISCRETE VARIABLES          8


GENERATION TIME      =        0.015 SECONDS      4 MB  24.1.1 r40636 WEX-WEI


EXECUTION TIME       =        0.015 SECONDS      4 MB  24.1.1 r40636 WEX-WEI
GAMS 24.1.1  r40636 Released May 30, 2013 WEX-WEI x86_64/MS Windows 12/22/20 08:35:15 Page 5
G e n e r a l   A l g e b r a i c   M o d e l i n g   S y s t e m
Solution Report     SOLVE ex1 Using MIP From line 36


               S O L V E      S U M M A R Y

     MODEL   ex1                 OBJECTIVE  cost
     TYPE    MIP                 DIRECTION  MINIMIZE
     SOLVER  CBC                 FROM LINE  36

**** SOLVER STATUS     1 Normal Completion         
**** MODEL STATUS      8 Integer Solution          
**** OBJECTIVE VALUE           736000.0000

 RESOURCE USAGE, LIMIT          0.015      1000.000
 ITERATION COUNT, LIMIT         0    2000000000

COIN-OR CBC      24.1.1 r40636 Released May 30, 2013 WEI x86_64/MS Windows    

Integer solution of 736000 found by feasibility pump after 0 iterations and 0 no
                                                              des (0.01 seconds)
Exiting as integer gap of 18949.93 less than 0 or 10%
Search completed - best objective 736000, took 0 iterations and 0 nodes (0.01 se
                                                                          conds)
Maximum depth 0, 0 variables fixed on reduced cost

Solved to optimality (within gap tolerances optca and optcr).
MIP solution:   7.360000e+005   (0 nodes, 0.015 seconds)
Best possible:  7.170501e+005
Absolute gap:   1.894993e+004   (absolute tolerance optca: 0)
Relative gap:   2.642762e-002   (relative tolerance optcr: 0.1)
Optimal - objective value 736000
```

#### Julia+JuMP实现

```julia
using JuMP
using Cbc
model = Model(Cbc.Optimizer)
month_num = 8
product_num = 1
month = 1:month_num
product = 1:product_num
# paramerters
q = [5000] # product fix cost
p = [100] # cost per product
d = [400, 400, 800, 800, 1200, 1200, 1200, 1200]' # demand of product by month
s_init = [200] # initial stock
h = [200] # stock cost per product month

@variable(model, x[i in product, j in month] >= 0)
@variable(model, s[i in product, j in month] >= 0)
@variable(model, y[i in product, j in month], Bin)

# 库存平衡
@constraint(model ,dem_sat1[i in product], s_init[i] + x[i, 1] == d[i, 1] + s[i, 1])
@constraint(model, dem_sat[i in product, j in month[2:end]], s[i, j - 1] + x[i, j] == d[i, j] + s[i, j] )
# 最大产量约束
@constraint(model, vub[i in product, j in month], x[i, j] <= sum(d[i, k]  for k in j:month_num) * y[i, j])
# 目标函数
@expression(model ,cost, sum(p[i] * x[i, j] + q[i] * y[i, j] for i in product, j in month) + sum(h[i] * s[i, j] for i in product, j in month[1:end - 1]) + sum(h[i] * s[i, month[end]] / 2 for i in product))
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

结果信息如下

```
Result - Optimal solution found

Objective value:                736000.00000000
Enumerated nodes:               0
Total iterations:               8
Time (CPU seconds):             0.29
Time (Wallclock seconds):       0.28

Total time (CPU seconds):       0.32   (Wallclock seconds):       0.32
```

#### Python+PuLP实现

```python
import pulp as pl
import numpy as np
# create model and solver
model = pl.LpProblem("1-1tiny-problem", pl.LpMinimize)
solver = pl.PULP_CBC_CMD()

month_num = 8
product_num = 1

# parameters
q = [5000] # product fix cost
p = [100] # cost per product
d = np.array([400, 400, 800, 800, 1200, 1200, 1200, 1200]).reshape((1,-1)) # demand of product by month
s_init = [200] # initial stock
h = [200] # stock cost per product month

# variables
x = pl.LpVariable.dicts('x', (range(product_num), range(month_num)), 0)
s = pl.LpVariable.dicts('s', (range(product_num), range(month_num)), 0)
y = pl.LpVariable.dicts('y', (range(product_num), range(month_num)), cat='Binary')

# objective
model += pl.lpSum(pl.lpSum(p[i]*x[i][j] + q[i]*y[i][j] \
            for i in range(product_num)) for j in range(month_num)) \
            + pl.lpSum(pl.lpSum(h[i]*s[i][j] for i in range(product_num)) \
             for j in range(month_num)) \
            + pl.lpSum(h[i]*s[i][month_num-1]/2 for i in range(product_num))

# constraints
for i in range(product_num):
    # init stock
    model += s_init[i] + x[i][0] == d[i][0] + s[i][0]
    # first month max limit of x
    model += x[i][0] <= pl.lpSum(d[i][k] for k in range(0, month_num))*y[i][0]
    for j in range(1, month_num):
        # stock balance
        model += s[i][j-1] + x[i][j] == d[i][j] + s[i][j]
        # max limit of x
        model += x[i][j] <= pl.lpSum(d[i][k] for k in range(j, month_num))*y[i][j]

result = model.solve(solver)
```

默认的结果如下

```
Result - Optimal solution found

Objective value:                736000.00000000
Enumerated nodes:               0
Total iterations:               14
Time (CPU seconds):             0.12
Time (Wallclock seconds):       0.12

Option for printingOptions changed from normal to all
Total time (CPU seconds):       0.20   (Wallclock seconds):       0.20
```

#### OR-Tools+CPP实现

```cpp
#include "ortools/linear_solver/linear_solver.h"
namespace operations_research
{
    struct DataModel
    {
        // parameter data
        // demand per product and month
        const std::vector<std::vector<double>> d{
            {400, 400, 800, 800, 1200, 1200, 1200, 1200},
        };
        // setup cost
        const std::vector<double> q{5000};
        // unit cost
        const std::vector<double> p{100};
        // unit stock cost
        const std::vector<double> h{5};
        // intial stock
        const std::vector<double> s_init{200};
        // product types
        const int num_product = 1;
        // months
        const int num_month = 8;
    };

    void MipVarArray()
    {
        DataModel data;
        // create an mip sovler
        MPSolver solver("integer_programming_example",
                        MPSolver::CBC_MIXED_INTEGER_PROGRAMMING);
        // define infinity
        const double infinity = solver.infinity();

        // declare x, s, y
        std::vector<std::vector<const MPVariable *>>
            x(data.num_product, std::vector<const MPVariable *>(data.num_month)),
            s(data.num_product, std::vector<const MPVariable *>(data.num_month)),
            y(data.num_product, std::vector<const MPVariable *>(data.num_month));

        for (int i = 0; i < data.num_product; ++i)
        {
            for (int j = 0; j < data.num_month; ++j)
            {
                x[i][j] = solver.MakeNumVar(0.0, infinity, "");
                s[i][j] = solver.MakeNumVar(0.0, infinity, "");
                y[i][j] = solver.MakeBoolVar("");
            }
        }

        // declare constraint
        for (int i = 0; i < data.num_product; ++i)
        {
            // constraint dem_sat month 1
            // s_init + x[i,1] = d[i, 1] + s[i, 1]
            MPConstraint *dem_sat1 = solver.MakeRowConstraint(
                data.d[i][0] - data.s_init[i], data.d[i][0] - data.s_init[i]);
            dem_sat1->SetCoefficient(x[i][0], 1);
            dem_sat1->SetCoefficient(s[i][0], -1);

            // constraint vub month 1
            // x[i, 1] <= sum(d[i, k] k>=1)*y[i, 1]
            MPConstraint *vub1 = solver.MakeRowConstraint(-infinity, 0);
            //coff_y = sum(d[i, k] k>=1)
            double coff_y = std::accumulate(data.d[i].begin(), data.d[i].end(), 0);
            vub1->SetCoefficient(x[i][0], 1);
            vub1->SetCoefficient(y[i][0], -coff_y);

            for (int j = 1; j < data.num_month; ++j)
            {
                // constraint dem_sat
                // s[i, j-1] + x[i, j] == d[i, j] + s[i, j]
                MPConstraint *dem_sat = solver.MakeRowConstraint(data.d[i][j], data.d[i][j]);
                dem_sat->SetCoefficient(x[i][j], 1);
                dem_sat->SetCoefficient(s[i][j], -1);
                dem_sat->SetCoefficient(s[i][j - 1], 1);

                // constraint vub
                // x[i, j] <= sum(d[i, k] k>=j)*y[i, j]
                MPConstraint *vub = solver.MakeRowConstraint(-infinity, 0);
                //coff_y = sum(d[i, k] k>=j)
                double coff_y = std::accumulate(data.d[i].begin() + j, data.d[i].end(), 0);
                vub->SetCoefficient(x[i][j], 1);
                vub->SetCoefficient(y[i][j], -coff_y);
            }
        }

        // objective cost
        // cost = sum(x[i, j]*q[i]) +sum(y[i, j]*p[i])
        //      + sum(s[i,j]*h[i] j<=8)+ sum(s[i, 8]*h/2)
        MPObjective *const cost = solver.MutableObjective();
        for (int i = 0; i < data.num_product; ++i)
        {
            for (int j = 0; j < data.num_month - 1; ++j)
            {
                cost->SetCoefficient(x[i][j], data.p[i]);
                cost->SetCoefficient(y[i][j], data.q[i]);
                cost->SetCoefficient(s[i][j], data.h[i]);
            }
            cost->SetCoefficient(x[i][data.num_month - 1], data.p[i]);
            cost->SetCoefficient(y[i][data.num_month - 1], data.q[i]);
            cost->SetCoefficient(s[i][data.num_month - 1], data.h[i] / 2);
        }

        cost->SetMinimization();

        const MPSolver::ResultStatus result_status = solver.Solve();

        // Check that the problem has an optimal solution.
        if (result_status != MPSolver::OPTIMAL)
        {
            LOG(FATAL) << "The problem does not have an optimal solution.";
        }
        LOG(INFO) << "Solution:";
        LOG(INFO) << "Optimal objective value = " << cost->Value();
        std::cout << "Optimal x value=";
        for (int i = 0; i < data.num_product; ++i)
        {
            std::cout << "\n";
            for (int j = 0; j < data.num_month; ++j)
            {
                std::cout << x[i][j]->solution_value() << ",";
            }
        }
        std::cout << "\n";
        std::cout << "Optimal y value=";
        for (int i = 0; i < data.num_product; ++i)
        {
            std::cout << "\n";
            for (int j = 0; j < data.num_month; ++j)
            {
                std::cout << y[i][j]->solution_value() << ",";
            }
        }
        std::cout << "\n";
        std::cout << "Optimal s value=";
        for (int i = 0; i < data.num_product; ++i)
        {
            std::cout << "\n";
            for (int j = 0; j < data.num_month; ++j)
            {
                std::cout << s[i][j]->solution_value() << ",";
            }
        }
    }
} // namespace operations_research

int main(int argc, char **argv)
{
    operations_research::MipVarArray();
    return EXIT_SUCCESS;
}
```

结果如下

```
I00-1 -1:-1:-1.246093 20680 1-1tiny-problem.cc:111] Solution:
I00-1 -1:-1:-1.586669 20680 1-1tiny-problem.cc:112] Optimal objective value = 736000
Optimal x value=
600,0,1600,0,1200,1200,1200,1200,
Optimal y value=
1,0,1,0,1,1,1,1,
Optimal s value=
400,0,800,0,0,0,0,0,
```

## 1.2 更贴近实际的一个生产计划案例

### 问题描述

#### GW 和 全球供应链优化部门

GW是一个大型快速消费品公司，面向全球数百万人销售数百种商品。Bill Widge是全球供应链优化（Global Supply Chain Optimization, GSCO）部门的领导，他负责开发、实现新的优化方法并将其集成到制造业信息系统中（著名的PASI-2系统）以提高产能利用率和流程灵活性。

该公司已经在所有工厂安装了一个通用的制造信息化、计划和控制系统（MPCS）。系统可以使公司计划和协调采购、生产和运输等环节。经过了数年的推行和定制化，系统为供应链协调带了了具体大的提升和改进。

但是，由于所有工厂采用的是同样的计划系统，计划程序是完全通用的针对于供应链系统的改善，难以提高制造工厂的生产效率（广义上说，指单位产出和投入比）和灵活性（指快速响应不断变化的市场需求）。

因此，董事会决定成立GSCO部门来弥补这一缺陷。GSCO的目标是在计划工具之上进行优化，以支持计划任务并提高关键流程的生产率。 最终目标是将针对这些过程的特定计划工具集成到通用信息系统PASI-2中。

**问题**

这里讨论的是GSCO成功的一个项目。 项目的目标是优化食品领域最大的工厂的生产率。该工厂生产两个系列的产品，称为谷物和水果，每个系列包含多种不同的产品。以下是关于项目的问题描述。

#### 生产流程

生产主要有三个部分组成，预处理、搅拌和包装

- 首先，仓储原料需要经过预处理（例如清洗、加热等）才能进入生产工段。预处理只有一条产线。
- 接着，谷物和水果将完成生产，这一工段主要是搅拌不同的配料因此被称为搅拌工段。搅拌工段还包括加热、粉碎、干燥等一些其他工作。搅拌只有一条产线。
- 最后，产品在两条不同的产线上完成包装。包装产线对于谷物和水果不能混用的。

![image-20201224004954478](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201224004954478.png)

生产流程在图中表示。请注意，对于构建和逐步微调模型，使用这种图形化流程表示问题中的元素有助于综合考虑信息，也是建模过程的重要组成部分。

#### 瓶颈

整个生产流程的瓶颈或者主要约束是搅拌工段，原因如下：

- 原料很少，采购期很短也很可靠。原料的库存量主要取决于其需求，受其他因素的影响很小。

- 预处理过程是非常迅速的连续过程，耗时很短，因此总能在当前批次搅拌的过程中完成下一批次的预处理过程。

- 搅拌工段的速度取决与搅拌操作/机器，搅拌工段的其他操作（加热、粉碎、干燥）可以在搅拌操作的同时完成，并不会影响整个工段的工作。

- 搅拌机器需要在每个批次结束需要固定的清洗耗时。为了保证产品质量，视频监管局要求清洗是必要的，而且不受工序或者产品类型影响。搅拌工段的其他操作没有清洗或者预处理耗时。除了清洗耗时，搅拌操作的每台机器生产速度可以被认为是恒定的，不会受生产规模影响。

- 得益于最近对第二条包装流水线的投资建设，现在每个系列的产品都有一条专门的包装流水线，可以在不同产品之间无缝切换。两条包装流水线总计产能可以满足搅拌工段的产能。

- 包装流水线是全自动化的，可以在同系列的产品之间几乎无缝切换。

- 最后，由于质量因素，在搅拌工段和包装工段之间只有非常有限的中间存储容量，产品几乎是在完成搅拌后直接进入包装工段。

综上，整个生产的瓶颈在于搅拌操作步骤，因为每个批次之后都搅拌机器都需要消耗较多时间完成清洗。预处理以及搅拌工段的其他操作都可以在搅拌操作的同步完成。包装工段的切换几乎没有额外耗时，而且有足够的产能（两条包装先的合计产能）接收搅拌完成的中间品。包装必须和搅拌同步，因为工段之间几乎没有中间存储容量。

#### 生产政策和现有的计划系统

由于产品种类有限，为了能缩短全球供应链的交货时间，公司在工厂一级实行MTS(make-to-stock)生产政策。MTS意思是工厂的生产必须要使库存能够满足配送系统的需求，也就是说在工厂一级的交货期为0。为了实现MTS政策，公司每周会对需求量进行预测，同时考虑到预测值和实际的偏差，还会给出一个安全库存量，此项工作在过去一直卓有成效，预测的准确性很高，而且安全库存的存在也为公司的服务添加了双保险。图1.4为添加了瓶颈和生产政策的相关信息的流程图。现有的计划管理系统是一个典型的ERP/MRP类型系统。

![image-20201224152136658](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201224152136658.png)

- 每周会生成一份主生产日程（Master Production Schedule MPS），包含最近几周的计划。日程计划在成品层级（包装层级）上需要满足预测需求量和安全库存量

- MPR系统会确定每个中间层级（预处理和搅拌）的产品的生产时间和生产量，原料的订购时间和订购量。MRP需要基于MRP系统上设定的成本批次顺序完成计算。

- 最后，会根据MRP系统对完成每周详细的包装和搅拌操作日程安排。


此处计划行业相关术语（ERP、MRP、MPS）在第二章有简要介绍。

现有的MPS/MRP系统使用一周作为规划的时间粒度;也就是说，时间被按周分解时间段。工厂每周运行五天，搅拌机必须于本周末进行清洁。因此，周末没有生产批次运行。选择一周的时间粒度是为了与预测系统一致。

MPS和MRP当前的使用时间范围为6周，略长于总采购和制造交货时间，以便GW能够及时订购和接收原材料。所有的原材料订单都可以根据主生产计划中的成本数量计算得到。

#### 挑战

然而，整个计划系统的核心部分MPS系统并不能将搅拌这一瓶颈操作作纳入到优化范围，因此计划人员需要人工修订MPS计划才能使其可行。该计划优化问题的主要难点在于权衡生产效率（批次生产量应尽量大以减少批次数量和清洗时间）和生产的灵活性（批次生产量应尽量小以快速响应市场需求）。

问题的根源在于MPS并不是由生产过程中最稀缺因素（搅拌产量）驱动，因此Bill Widge和GSCO部门需要设计开发完成一个可以给出同时满足需求和产能利用率的可行的生产计划的MPS工具。主要目标是尽可能的推迟生产计划以改善生产的灵活性。

### 建模

在我们使用逐步法进行建模时，我们需要确定模型的范围，确定模型宇宙的边界（考虑什么产品？建模什么资源？），确定模型的一般结构。模型的详细程度也是一个主要的决策，反映实际优化中的生产效率和灵活性的权衡必须要有足够细致程度，但是过于详细的模型也会导致模型不能求解或者过于困难。

在前一个示例中，泛型约束（指一类约束可以使用一个约束名称来代替，例如[第一个案例](## 1.1 自行车生产的案例)中的dem_sat表示了8个约束，在本案例中同样存在)的作用变得清晰。我们能够重用前面遇到的一些约束，从而显著简化建模任务。

首先需要根据问题描述来对索引、参数、变量和约束进行定义和命名，构建模型的要素。

#### 对象集合

| 对象和索引           | 对象数学表示 | 索引数学表示      |
| -------------------- | ------------ | ----------------- |
| 一个工厂             | --           |                   |
| 产品系列C和F         | --           |                   |
| 不同的成品           | products     | i= 1,..., NI      |
| 时间段（周）         | periods      | t=1,...,NT ,NT=15 |
| 混合流水线           | machine      | k=1               |
| 谷物包装线           | machine      | k=2               |
| 水果包装线           | machine      | k=3               |
| 其他产品、资源：忽略 | --           |                   |
| 库存资源：忽略       | --           |                   |

备注和假设

- MPS模型必须独立考虑每种成品（或者使搅拌后的中间产品，因为它与成品一一对应），才能满足产品需求预测。
- 仍然使用一周的时间粒度，因为没有必要去提高或者降低时间力度。
- 为了优化产能利用率有必要在足够长的一个时间段内预测产能需求，MPS考虑的时间长度需要比总采购和制造提前期（约六周）长得多。根据对需求短期变化的更深入的分析，选择15周（超参）。
- 由于清洗耗时，搅拌操作的产能是整个过程的主要的瓶颈，当也需要考虑到包装线的产能，因为单独一条包装线不能覆盖所有的搅拌产能。
- 除了搅拌和包装以外的其他操作都被忽略，因为这些操作都可以在搅拌过程中同步完成且并不会带来附加的限制。

#### 参数

| 名称               | 符号     | 索引 | 值   |
| ------------------ | -------- | ---- | ---- |
| 预测需求           | d        | i, t |      |
| 安全库存           | SS       | i,t  |      |
| 初始库存           | ss_init  | i    |      |
| 搅拌后的清洗时间   | $\beta$  | i    |      |
| 单位产品的生产耗时 | $\alpha$ | i, k |      |
| 机器产能           | L        | k    |      |
| 谷物类产品（子集） | F2       |      |      |
| 水果类产品（子集） | F3       |      |      |

备注和假设

- 假设批次搅拌操作后的清洗耗时取决于产品的种类
- 为了利用搅拌和包装的产能，还需要明确每个周期中每台机器的运行时间
- 最后，产品所属的系列（谷物、水果）需要已知，以便将搅拌批次产品合理的安排给包装线，同时考虑包装线的容量限制。

参数数据

需求预测数据

|      | t1   | t2   | t3   | t4   | t5   | t6   | t7   | t8   | t9   | t10  | t11  | t12  | t13  | t14  | t15  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| i1   | 0    | 95   | 110  | 96   | 86   | 124  | 83   | 108  | 114  | 121  | 110  | 124  | 104  | 86   | 87   |
| i2   | 98   | 96   | 96   | 98   | 103  | 104  | 122  | 101  | 89   | 108  | 101  | 109  | 106  | 108  | 76   |
| i3   | 106  | 0    | 89   | 123  | 96   | 105  | 83   | 82   | 112  | 109  | 119  | 85   | 99   | 80   | 123  |
| i4   | 98   | 121  | 0    | 105  | 98   | 96   | 101  | 81   | 117  | 76   | 103  | 81   | 95   | 105  | 102  |
| i5   | 0    | 124  | 113  | 123  | 123  | 79   | 111  | 98   | 97   | 80   | 98   | 124  | 78   | 108  | 109  |
| i6   | 103  | 102  | 0    | 95   | 107  | 105  | 107  | 105  | 75   | 93   | 115  | 113  | 111  | 105  | 85   |
| i7   | 110  | 93   | 0    | 112  | 84   | 124  | 98   | 101  | 83   | 87   | 105  | 118  | 115  | 106  | 78   |
| i8   | 85   | 92   | 101  | 110  | 93   | 96   | 120  | 109  | 121  | 87   | 92   | 85   | 91   | 93   | 109  |
| i9   | 122  | 116  | 109  | 0    | 105  | 108  | 88   | 98   | 77   | 90   | 110  | 102  | 107  | 99   | 96   |
| i10  | 120  | 124  | 94   | 105  | 92   | 86   | 101  | 106  | 75   | 109  | 83   | 95   | 79   | 108  | 100  |
| i11  | 117  | 96   | 78   | 0    | 108  | 87   | 114  | 107  | 110  | 94   | 104  | 101  | 108  | 110  | 80   |
| i12  | 125  | 112  | 75   | 0    | 116  | 103  | 122  | 88   | 85   | 84   | 76   | 102  | 84   | 88   | 82   |

|      | F产品包装机器 | ss_init初始库存 | SS安全库存 | $\alpha$生产耗时 | $\beta$清洗时间 |
| ---- | ------------- | --------------- | ---------- | ---------------- | --------------- |
| i1   | 2             | 83              | 10         | 1                | 30              |
| i2   | 2             | 31              | 10         | 1                | 20              |
| i3   | 2             | 11              | 10         | 1                | 30              |
| i4   | 2             | 93              | 10         | 1                | 40              |
| i5   | 2             | 82              | 10         | 1                | 40              |
| i6   | 2             | 72              | 10         | 1                | 10              |
| i7   | 3             | 23              | 20         | 1                | 30              |
| i8   | 3             | 91              | 20         | 1                | 20              |
| i9   | 3             | 83              | 20         | 1                | 10              |
| i10  | 3             | 34              | 20         | 1                | 50              |
| i11  | 3             | 61              | 20         | 1                | 30              |
| i12  | 3             | 82              | 20         | 1                | 20              |

机器生产限制

|      | L    |
| ---- | ---- |
| k1   | 1400 |
| k2   | 700  |
| k3   | 700  |

#### 变量

| 名称                         | 符号 | 索引 | 类型     |
| ---------------------------- | ---- | ---- | -------- |
| 搅拌操作处理量（等于成品量） | x    | i, t | Positive |
| 生产开停工                   | y    | i, t | Binary   |
| 周期结束的库存量             | s    | i, t | Positive |

备注和假设

- 主要的决策变量是在每个时间粒度内每种产品搅拌操作处理量，也就是成品量
- 使用生产开停工来表示是否需要计算机器清洗耗时，在一个周期内一种产品只会有一次生产
- 为了权衡生产效率和生产灵活性，成品库存也是模型需要考虑的因素

#### 约束

| 名称               | 符号      | 索引     |
| ------------------ | --------- | -------- |
| 生产满足需求       | dem_sat   | i, t     |
| 安全库存约束       | ss_limit  | i, t     |
| 生产上限           | vub       | i, t     |
| 搅拌机器的运行上限 | mix_capt  | t        |
| 包装机器的运行上限 | pack_capt | t,k(k>1) |

备注和假设

- 生产上限约束与[第一个案例](##1.1自行车生产的案例)一样
- 搅拌和包装应当在同一个时间段内完成，因为并没有中间存储

#### 目标

| 名称     | 符号      |
| -------- | --------- |
| 库存总量 | inventory |

备注和假设

- 模型的目标是尽可能的推迟生产操作，也可以理解为是成品库存量总和最小

#### 模型结构

- 15个时间段内，成品的需求量约束，包括初始库存量和安全库存量
- 每个时间段内，搅拌机器的处理量上限约束，包括清洗的时间
- 每个时间段内，包装机器的处理量上限约束
- 最小库存总和为目标

![image-20201224221628446](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201224221628446.png)

### 数学模型

接着对模型目标和约束进行数学表示

优化目标inventory 库存的总和最小
$$
inventroy = \sum_{t}\sum_{i}s_{i, t}
$$
约束：

- dem_sat  生产+上月库存=需求+下月库存
  $$
  s_{i, t-1}+x_{i, t} = d_{i,t}+s_{i,t}
  $$

- ss_limit 库存应大于安全库存
  $$
  s_{i, t} \geq SS_{i, t}
  $$

- vub 每月的生产要小于后续所有的需求量和安全库存量之和
  $$
  x_{i, t} \leq \left((\sum_{k=t}^{NT}D_{i, k}) +SS_{i, NT}\right)y_{i, t}
  $$

- mix_cap 搅拌机器每月运行上限
  $$
  \sum_i\alpha_{i, 1}x_{i,t}+\sum_i\beta_{i}y_{i, t}\leq L_{1}
  $$

- pack_cap 包装机器每月运行上限
  $$
  \sum_{i\in F_k}\alpha_{i,k}x_{i,t} \leq L_k,\quad k=2,3
  $$

### 模型实现

#### Julia实现

```Julia
using JuMP

# use gurobi
using Gurobi
model = Model(Gurobi.Optimizer)
set_optimizer_attribute(model, "Cuts", 0) # 注释不启用cuts
set_optimizer_attribute(model, "TimeLimit", 60)

# uncomment to switch to cbc
# using Cbc
# model = Model(Cbc.Optimizer)
# set_optimizer_attribute(model, "cuts","off")
# set_optimizer_attribute(model, "seconds", 60)
# set_optimizer_attribute(model, "thread", 16)

n_i = 12                # 产品种类个数
n_t = 15                # 时间周期数
n_machine = 3           # 机器个数
machine = 1:n_machine   # 机器编号
product = 1:n_i         # 产品编号
weeks = 1:n_t           # 时间编号
paras = [   2 83 10 1 30;
            2 31 10 1 20;
            2 11 10 1 30;
            2 93 10 1 40;
            2 82 10 1 40;
            2 72 10 1 10;
            3 23 20 1 30;
            3 91 20 1 20;
            3 83 20 1 10;
            3 34 20 1 50;
            3 61 20 1 30;
            3 82 20 1 20]

f = paras[:,1] # 产品包装使用的机器编号、产品类别

f2 = product[f.==2] # 谷物类产品
f3 = product[f.==3] # 水果类产品

s_init = paras[:,2] # 初始库存
ss = paras[:, 3] # 安全库存
alpha = repeat(paras[:, 4], 1,n_machine) # 生产速度
beta = paras[:, 5] # 清洗时间
# 需求
d  =[   0 95 110 96 86 124 83 108 114 121 110 124 104 86 87;
        98 96 96 98 103 104 122 101 89 108 101 109 106 108 76;
        106 0 89 123 96 105 83 82 112 109 119 85 99 80 123;
        98 121 0 105 98 96 101 81 117 76 103 81 95 105 102;
        0 124 113 123 123 79 111 98 97 80 98 124 78 108 109;
        103 102 0 95 107 105 107 105 75 93 115 113 111 105 85;
        110 93 0 112 84 124 98 101 83 87 105 118 115 106 78;
        85 92 101 110 93 96 120 109 121 87 92 85 91 93 109;
        122 116 109 0 105 108 88 98 77 90 110 102 107 99 96;
        120 124 94 105 92 86 101 106 75 109 83 95 79 108 100;
        117 96 78 0 108 87 114 107 110 94 104 101 108 110 80;
        125 112 75 0 116 103 122 88 85 84 76 102 84 88 82]
L = [1400, 700, 700] # 机器上限

@variable(model, x[i in product, t in weeks] >= 0) # 产品生产计划
@variable(model, s[i in product, t in weeks] >= 0) # 库存
@variable(model, y[i in product, t in weeks], Bin) # 是否开工生产

# 库存平衡
@constraint(model ,dem_sat1[i in product], s_init[i] + x[i, 1] == d[i, 1] + s[i, 1])
@constraint(model, dem_sat[i in product, t in weeks[2:end]], s[i, t - 1] + x[i, t] == d[i, t] + s[i, t] )
# 安全库存
@constraint(model, ss_limit[i in product, t in weeks], s[i, t] >= ss[i])

# 最大产量约束
@constraint(model, vub[i in product, t in weeks], x[i, t] <= (sum(d[i, k]  for k in t:n_t) + ss[i]) * y[i, t])

# 混合机器上限
@constraint(model, mix_cap[t in weeks], sum(alpha[i, 1]*x[i, t] for i in product) + sum(beta[i]*y[i, t] for i in product) <=L[1])
# 包装机器上限
@constraint(model, pack_cap2[t in weeks], sum(alpha[i, 2]*x[i,t] for i in f2) <= L[2])
@constraint(model, pack_cap3[t in weeks], sum(alpha[i, 3]*x[i,t] for i in f3) <= L[3])
@expression(model, inventory, sum(s[i, t] for i in product, t in weeks))
@objective(model, Min, inventory)

optimize!(model)
```

### 优化结果

### 最优解

使用Gurobi 9.11或者SCIP7.0.1 默认的参数求解，可以找到该问题的全局最优解。

报告如下

```
Objective: 5730
Gap: 0.0038%
Nodes: 1449794
Time cost: 201.88
Solver: Gurobi 9.1.1 with JuMP
CPU: AMD 4800HS 8C16T
```

```
Objective: 5730
Gap: 0.00%
Nodes: 717912
Time cost: 784.00
Solver: SCIP 7.0.1 with JuMP
CPU: AMD 4800HS 8C16T
```

需要说明的是，Gurobi约在1秒内就找到了5730最优解，SCIP大约在40秒之后才找到，剩余时间均为求解器在遍历分支。

x_optimal

|      | t1   | t2   | t3   | t4   | t5   | t6   | t7   | t8   | t9   | t10  | t11  | t12  | t13  | t14  | t15  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| i1   | 0    | 22   | 161  | 131  | 0    | 207  | 0    | 108  | 114  | 151  | 80   | 127  | 139  | 105  | 30   |
| i2   | 77   | 96   | 96   | 99   | 102  | 162  | 64   | 101  | 89   | 108  | 101  | 109  | 106  | 108  | 76   |
| i3   | 105  | 0    | 89   | 219  | 0    | 105  | 165  | 0    | 112  | 109  | 204  | 0    | 99   | 80   | 123  |
| i4   | 15   | 121  | 105  | 0    | 194  | 0    | 182  | 0    | 193  | 0    | 184  | 0    | 200  | 0    | 102  |
| i5   | 0    | 52   | 113  | 123  | 202  | 0    | 111  | 195  | 0    | 178  | 0    | 202  | 0    | 108  | 109  |
| i6   | 41   | 102  | 0    | 95   | 107  | 105  | 107  | 180  | 0    | 93   | 115  | 113  | 111  | 105  | 85   |
| i7   | 107  | 93   | 0    | 196  | 0    | 222  | 0    | 101  | 170  | 0    | 105  | 118  | 115  | 106  | 78   |
| i8   | 14   | 92   | 101  | 110  | 189  | 0    | 120  | 109  | 121  | 87   | 177  | 0    | 91   | 93   | 109  |
| i9   | 59   | 116  | 109  | 0    | 105  | 108  | 88   | 98   | 77   | 90   | 110  | 102  | 107  | 99   | 96   |
| i10  | 106  | 124  | 94   | 197  | 0    | 187  | 0    | 181  | 0    | 192  | 0    | 174  | 0    | 108  | 100  |
| i11  | 76   | 96   | 78   | 0    | 195  | 0    | 221  | 0    | 204  | 0    | 104  | 101  | 108  | 110  | 80   |
| i12  | 63   | 112  | 75   | 0    | 116  | 103  | 122  | 88   | 85   | 160  | 0    | 102  | 84   | 88   | 82   |

s_optimal

|      | t1   | t2   | t3   | t4   | t5   | t6   | t7   | t8   | t9   | t10  | t11  | t12  | t13  | t14  | t15  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| i1   | 83   | 10   | 61   | 96   | 10   | 93   | 10   | 10   | 10   | 40   | 10   | 13   | 48   | 67   | 10   |
| i2   | 10   | 10   | 10   | 11   | 10   | 68   | 10   | 10   | 10   | 10   | 10   | 10   | 10   | 10   | 10   |
| i3   | 10   | 10   | 10   | 106  | 10   | 10   | 92   | 10   | 10   | 10   | 95   | 10   | 10   | 10   | 10   |
| i4   | 10   | 10   | 115  | 10   | 106  | 10   | 91   | 10   | 86   | 10   | 91   | 10   | 115  | 10   | 10   |
| i5   | 82   | 10   | 10   | 10   | 89   | 10   | 10   | 107  | 10   | 108  | 10   | 88   | 10   | 10   | 10   |
| i6   | 10   | 10   | 10   | 10   | 10   | 10   | 10   | 85   | 10   | 10   | 10   | 10   | 10   | 10   | 10   |
| i7   | 20   | 20   | 20   | 104  | 20   | 118  | 20   | 20   | 107  | 20   | 20   | 20   | 20   | 20   | 20   |
| i8   | 20   | 20   | 20   | 20   | 116  | 20   | 20   | 20   | 20   | 20   | 105  | 20   | 20   | 20   | 20   |
| i9   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   |
| i10  | 20   | 20   | 20   | 112  | 20   | 121  | 20   | 95   | 20   | 103  | 20   | 99   | 20   | 20   | 20   |
| i11  | 20   | 20   | 20   | 20   | 107  | 20   | 127  | 20   | 114  | 20   | 20   | 20   | 20   | 20   | 20   |
| i12  | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 20   | 96   | 20   | 20   | 20   | 20   | 20   |

### 初次接触算法

通常来说，MIP算法主要有两个部分，分支定界(Branch and Bound)和Cut Plane，详细的内容将在第3章介绍，这里只提及简略要点。

- 分支定界算法通过对可行空间的剪枝来减小问题规模，它依赖于线性松弛算法来寻找可行解，线性松弛算法就是将整数约束松弛为线性约束的方法，$y\in \{0, 1\} \rightarrow 0\leq y\leq 1 $

- 利用线性松弛，可以将MIP问题转换为LP问题，LP问题一般来说是容易求解的，但是LP问题的解不能保证满足整数约束（LP的解一定约束边界上，但是不一定是$y\in \{0, 1\}$）

- LP问题不满足整数约束的情况下，LP问题可以为MIP问题提供优化目标下限（lower bound ），MIP问题的最优目标不可能低于LP问题的目标值。

- B&B算法通过枚举一系列线性松弛条件，依据线性松弛是否能够满足整数约束来对解空间进行划分，换句话说就是对不能线性松弛的约束进行分支。同时，根据所有线性松弛获得解的下限。这就是算法的分支部分（branch）

- 在枚举过程中，会生成模型的一些可行解，即松弛整数变量采用整数值的解。 每个此类可行解决方案的目标值都提供了最佳目标值的上限（upper bound）。 这是算法的定界部分（bound）。

- 最后，如果枚举完成，则分支定界算法是精确的；如果枚举被截断，则仅能提供近似或启发式的可行解，通常使用对偶间隙（duality gap）来评价可行解的质量
  $$
  \text{Duality Gap} = \frac{\text{Best UB} - \text{Best LB}}{\text{Best LB}}\times 100\%
  $$
  式中Best LB 和 Best UB是整个算法遍历过程中最佳的lower bound和 uppper bound。全局最优解一定在LB和UB之间，DualityGap越小，则可行解离全局最优解也越近。Duality Gap的标准定义以Best UB做基准，不过大部分求解器的结果都是以Best LB为基准。

- Cut Plane算法在这里的作用就是将线性松弛后不能满足的整数约束的约束，修饰成更小范围的线性约束，从而减少B&B问题的规模。

通常来说，B&B算法的运行时间（主要是枚举过程中的线性求解时间，可以认为近似与线性求解调用次数相关）或者说在给定时间内的求解质量，与算法的分支数量直接相关。这也就是为什么要在MIP求解过程中引入CutPlane算法的原因，减少分支的数量来提高求解效率。

下面我们比较在同一求解器下CutPlane算法对结果影响。以下测试均限定求解时间为60S(测试CPU为AMD 4800HS 8C16T，根据CPU速度可自行设置)，采用求解器默认参数，除了`Cuts`是否启用不同。

首先使用Gurobi 测试，不注释第4行，使用Cut-Plane算法；注释改行，则不启用Cut-Plane算法

```julia
set_optimizer_attribute(model, "Cuts", 0) # 注释不启用cuts
```

结果如下

| Option   | Best Objective | Nodes   | Best Bound | Gap      |
| -------- | -------------- | ------- | ---------- | -------- |
| Cuts:off | 5732           | 2706841 | 3545       | 38.1614% |
| Cuts:on  | 5730           | 412143  | 5654       | 1.3101%  |

我们看到得益于Gurobi强大的能力，即便不启用Cut-Plane算法求解器也通过较好的Presolve确定了良好的初始解，几乎找到了最优解，但是不采用CutPlane 的Gap很大。

当我们切换到Cbc，似乎更容易看到两者差距，将Gurobi相关代码注释，并使用如下Cbc求解器设置（Cbc默认不会启用全部线程，需手动设置）

```julia
using Cbc
model = Model(Cbc.Optimizer)
set_optimizer_attribute(model, "cuts","off")
set_optimizer_attribute(model, "seconds", 60)
set_optimizer_attribute(model, "thread", 16)
```

| Option   | Best Objective | Nodes | Best Bound | Gap  |
| -------- | -------------- | ----- | ---------- | ---- |
| Cuts:off | 6706           | 38034 | 3186       | 110% |
| Cuts:on  | 5740           | 84725 | 5563       | 3%   |

对于Cbc来说，CutPlane算法效果很显著，因为在有限的时间内，CutPlane减少了大量无效的分支规模。

改善或收紧约束不是一件简单的事情。 我们将在第4章中展示如何使用开发分类和方案制定来改善结果，并迅速获得较优解，或者在合理的时间内获得可证明的最佳解。

### 备注

## 练习

### 练习 1.1

考虑1.1的排产问题，新增了第二种型号（山地车）的自行车产品。问题排产范围仍然是Jan-Aug。第二种自行车的预测需求均为200，除了7月和8月是500；初始的库存为0；固定生产费用为3000，单位生产费用为60，单位库存费用为3。

1. 采用同样的MIP模型，计算山地车的优化方案
2. 新增一条约束，每月所有自行车的产量最高为1500，同时优化两种自行车的生产方案
3. 分析以下产量约束对问题的影响

### 练习1.2

在GW的案例中，如果每种产品的包装线每个时间段只能包装一种产品该如何修改模型。

这种类型的约束，被称为生产模式约束，可以简化搅拌器的安排减少清洁费用，但是会减少产线的灵活性。我们假设在这里使用比原始问题更小的时间段，按照天计算而不是星期。

### 练习1.3

在GW的案例中，如果每个时间段内只能搅拌一种产品，模型改如何修改？

同样使用天作为时间段。在现有的情况下，修改的模型是否有意义？

### 练习1.4

考虑GW的问题，如果在两个星期的连续生产中，上一周生产的最后一批产品和下一周生产的第一批产品相同，则可以不用清洗机器。

1. 修改MIP模型，将清洗的扣除考虑到其中
2. 求解该MIP模型，是否比远逝问题更难求解

提示：考虑新增变量，以及新变量和是否生产变量之间的关系

### 练习1.5

考虑GW问题，如果清洗的时间与产品的顺序有关。具体来说，清洗时间取决于之后搅拌生产的产品。搅拌机每周结束会清洗一次，清洗时间可以视为一种虚拟产品。

1. 建立MIP模型
2. 尝试创建一组清洗时间的顺序，并求解MIP模型。

### 练习1.6

在GW问题中，如果搅拌和混合操作之间可以存储，你会如何修改模型

1. 假设优化目标为总的库存量（搅拌后和混合后的库存）最小，尝试修改MIP模型
2. 尝试求解MIP模型，这比原始问题更加困难吗，困难点在哪里？







