# SCIP

# CBC

## General

| 参数         | 值   | 描述                      |
| ------------ | ---- | ------------------------- |
| seconds      | Int  | 终止时间                  |
| logLevel     | Int  | 输出日志等级，0不输出日志 |
| maxSolutions | Int  | 保存可行解个数，默认为1   |
| maxNodes     | Int  | 终止最大节点个数          |
| allowableGap | R+   | 终止Gap绝对值             |
| ratioGap     | 0-1  | 终止Gap相对值             |
| thread       | Int  | 线程数                    |

## CutGeneration

| 参数 | 值      | 描述               |
| ---- | ------- | ------------------ |
| cuts | on, off | 启用cut egenration |
|      |         |                    |
|      |         |                    |



# Gurobi

| 参数       | 值   | 描述                |
| ---------- | ---- | ------------------- |
| TimeLimit  | Int  | 终止时间            |
| Presolve   | 0, 1 | 启用预求解          |
| Heuristics | 0, 1 | 启用启发式          |
| Cuts       | 0, 1 | 启用cuts generation |

