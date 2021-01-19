import pulp as pl
import numpy as np
# create model and solver
model = pl.LpProblem("1-1tiny-problem", pl.LpMinimize)

month_num = 8
product_num = 1

# parameters
q = [5000]  # product fix cost
p = [100]  # cost per product
d = np.array([400, 400, 800, 800, 1200, 1200, 1200, 1200]).reshape(
    (1, -1))  # demand of product by month
s_init = [200]  # initial stock
h = [5]  # stock cost per product month

# variables
x = pl.LpVariable.dicts('x', (range(product_num), range(month_num)), 0)
s = pl.LpVariable.dicts('s', (range(product_num), range(month_num)), 0)
y = pl.LpVariable.dicts('y', (range(product_num), range(month_num)),
                        cat='Binary')

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
    model += x[i][0] <= pl.lpSum(d[i][k]
                                 for k in range(0, month_num)) * y[i][0]
    for j in range(1, month_num):
        # stock balance
        model += s[i][j - 1] + x[i][j] == d[i][j] + s[i][j]
        # max limit of x
        model += x[i][j] <= pl.lpSum(d[i][k]
                                     for k in range(j, month_num)) * y[i][j]

# solver = pl.PULP_CBC_CMD()
# solver = pl.GUROBI_CMD(path=r"C:\gurobi911\win64\bin\gurobi_cl.exe")
solver = pl.SCIP_CMD(path=r"C:\Program Files\SCIPOptSuite 7.0.2\bin\scip.exe")
# solver = pl.GLPK_CMD(path=r"C:\Personal\Binary\glpk465\glpsol.exe")
# solver = pl.MOSEK()
# solver = pl.GUROBI()
result = model.solve(solver)
print("x val is")
for i in range(product_num):
    for j in range(month_num):
        print(pl.value(x[i][j]), end=",")
    print("\n")
