using JuMP
using MathOptInterface
const MOI = MathOptInterface

# using Gurobi
# model = Model(Gurobi.Optimizer)
# set_optimizer_attribute(model, "Cuts",0)
# set_optimizer_attribute(model, "TimeLimit", 60)

# using MosekTools
# model = Model(Mosek.Optimizer)
# set_optimizer_attribute(model,"MSK_DPAR_MIO_MAX_TIME", 60)

# using SCIP
# model = Model(SCIP.Optimizer)
# set_optimizer_attribute(model,"limits/time", 60)


using Cbc
model = Model(Cbc.Optimizer)
set_optimizer_attribute(model, "cuts","off")
set_optimizer_attribute(model, "seconds", 60)
set_optimizer_attribute(model, "thread", 16)

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

f2 = product[f .== 2] # 谷物类产品
f3 = product[f .== 3] # 水果类产品

s_init = paras[:,2] # 初始库存
ss = paras[:, 3] # 安全库存
alpha = repeat(paras[:, 4], 1, n_machine) # 生产速度
beta = paras[:, 5] # 清洗时间
# 需求
d  = [   0 95 110 96 86 124 83 108 114 121 110 124 104 86 87;
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
@constraint(model, mix_cap[t in weeks], sum(alpha[i, 1] * x[i, t] for i in product) + sum(beta[i] * y[i, t] for i in product) <= L[1])
# 包装机器上限
@constraint(model, pack_cap2[t in weeks], sum(alpha[i, 2] * x[i,t] for i in f2) <= L[2])
@constraint(model, pack_cap3[t in weeks], sum(alpha[i, 3] * x[i,t] for i in f3) <= L[3])
@expression(model, inventory, sum(s[i, t] for i in product, t in weeks))
@objective(model, Min, inventory)

optimize!(model)
