using LinearAlgebra
using JuMP
using Cbc

model = Model(Cbc.Optimizer)
set_optimizer_attribute(model, "cuts","off")
# set_optimizer_attribute(model, "msg_lev", GLPK.GLP_MSG_ON)
# set_optimizer_attribute(model, "maxNodes", 5)
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
