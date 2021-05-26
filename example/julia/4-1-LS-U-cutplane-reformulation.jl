using JuMP
using Cbc
model = Model(Cbc.Optimizer)
# set_optimizer_attribute(model, "cuts","off")

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