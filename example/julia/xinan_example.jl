using JuMP
using Cbc
using SCIP
using Clp
d = zeros(6, 4);
sumd = zeros(6, 4);
orderquantity = [500, 800, 1500, 700, 900, 500];
order = [1,2,3,4,5,6]

ordertime = [3, 4, 5, 4, 5, 6]
orderitem = [1, 1, 1, 2, 2, 1]

d[3, 1] = 500
d[4, 1] = 800
d[5, 1] = 1500
d[4, 2] = 700
d[5, 2] = 900
d[6, 1] = 500

mps = Model(Cbc.Optimizer)
items = [1,2,3,4]
date = [1,2,3,4,5,6]
res = [1]
s_init = 0
alpha = [20, 30, 0, 0]
@variable(mps, x[t in date,i in items,] >= 0)
@variable(mps, s[t in date, i in items] >= 0)
@variable(mps, y[t in date, i in items], Bin)

@constraint(mps, dem_sat1[i in items], s_init + x[1, i] == d[1, i] + s[1, i])
@constraint(mps, dem_sat[t in date[2:end], i in items], s[ t - 1, i] + x[t, i] == d[t, i] + s[t, i])
@constraint(mps, vub[t in date, i in items], x[t, i] <= sum(d[k, i]  for k in date[t:end]) * y[t, i])
@constraint(mps, reslimit[t in date], sum(x[t,i] * alpha[i] for i in items) <= 3600 * 8)
@objective(mps, Min, sum(s[t, i] for i in items, t in date))
optimize!(mps)
x_val = value.(x)
x_val[2,1] = 460
x_val[3,1] = 840

sub1 = Model(Clp.Optimizer)
order1 = [1,2,3,6]
@variable(sub1, wx[t in date, p in order1] >= 0)
# @variable(sub1, wy[t in date, p in order1], Bin)
@constraint(sub1,  dem_order[p in order1], sum(wx[t, p] for t in 1:ordertime[p])  == orderquantity[p] )
# @constraint(sub1, xlimit[t in date, p in order1], wx[t, p] <= x_val[t,1] );
@constraint(sub1, xeq[t in date], sum(wx[t, p] for p in order1) == x_val[ t,1]);
@objective(sub1, Min, sum(wx[t, p] * (ordertime[p] - 1) for t in date, p in order1));
optimize!(sub1);
@show value.(wx);

sub2 = Model(Clp.Optimizer)
order2 = [4,5]
@variable(sub2, wx2[t in date, p in order2] >= 0)
@constraint(sub2 , dem_order2[p in order2], sum(wx2[t, p] for t in 1:ordertime[p]) == orderquantity[p])
@constraint(sub2, xeq2[t in date], sum(wx2[t, p] for p in order2) == x_val[t,2])

@objective(sub2, Min, sum(wx2[t, p] * (ordertime[p] - 1) for t in date, p in order2) )
optimize!(sub2)
@show value.(wx2)

sub0 =  Model(Clp.Optimizer)
@variable(sub0, wx0[t in date, p in order] >= 0)
@constraint(sub0, dem_order0[p in order], sum(wx0[t, p] for t in 1:ordertime[p]) == orderquantity[p])
@constraint(sub0, xeq01[t in date], sum(wx0[t, p] for p in order1) == x_val[t, 1])
@constraint(sub0, xeq02[t in date], sum(wx0[t, p] for p in order2) == x_val[t, 2])

@objective(sub0, Min, sum(wx0[t, p] * (ordertime[p] - 1) for t in date, p in order))
optimize!(sub0)
@show value.(wx0)