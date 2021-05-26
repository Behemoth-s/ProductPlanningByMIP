using JuMP
using Cbc
using SCIP
d = zeros(6, 4);
sumd = zeros(6, 4);

d[3,1] =  1300
d[5,1] = 1500
d[6, 1] = 500
d[3,2] = 400
d[4,2] = 1200

mps = Model(Cbc.Optimizer)
items = [1,2,3,4]
date = [1,2,3,4,5,6]
res = [1]
s_init = 0
@variable(mps, x[t in date,i in items,] >= 0)
@variable(mps, s[t in date, i in items] >= 0)
@variable(mps, y[t in date, i in items], Bin)

@constraint(mps, dem_sat1[i in items], s_init + x[1, i] == d[1, i] + s[1, i])
@constraint(mps, dem_sat[t in date[2:end], i in items], s[ t - 1, i] + x[t, i] == d[t, i] + s[t, i])
@constraint(mps, vub[t in date, i in items], x[t, i] <= sum(d[k, i]  for k in date[t:end]) * y[t, i])
@objective(mps, Min, sum(s[t, i] for i in items, t in date))
optimize!(mps)
x_val = value.(x)
x_val[2,1] = 460
x_val[3,1] = 840
order = [1,2,3,4,5,6]
orderquantity = [500, 800, 1500, 700, 900, 500];
ordertime = [3, 3, 5, 4, 5, 6]

sub1 = Model(Cbc.Optimizer)
order1 = [1,2,3,6]
@variable(sub1, wx[t in date, p in order1] >= 0)
@variable(sub1, wy[t in date, p in order1], Bin)
@constraint(sub1,  dem_order[p in order1], sum(wx[t, p] for t in 1:ordertime[p])  == orderquantity[p] )
@constraint(sub1, xlimit[t in date, p in order1], wx[t, p] <= x_val[t,1] * wy[t, p]);
@constraint(sub1, xeq[t in date], sum(wx[t, p] for p in order1) == x_val[ t,1]);
@objective(sub1, Min, sum(wy[t, p] * ordertime[p] for t in date, p in order1));
optimize!(sub1);
@show value.(wx);
