using JuMP
using Cbc

model = Model(Cbc.Optimizer)
# set_optimizer_attribute(model, "msg_lev", GLPK.GLP_MSG_ON)
# set_optimizer_attribute(model, "maxNodes", 5)

month_num = 8
product_num = 1
month = 1:month_num
product = 1:product_num

# paramerters
q = [5000] # product fix cost
p = [100] # cost per product
d = [400, 400, 800, 800, 1200, 1200, 1200, 1200]' # demand of product by month
s_init = [200] # initial stock
h = [5] # stock cost per product month

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
