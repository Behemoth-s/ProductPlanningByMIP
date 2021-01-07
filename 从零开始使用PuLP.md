# 从零开始的PuLP教程

[PuLP](https://github.com/coin-or/pulp)是一个求解LP/MIP问题的开源python建模软件包。PuLP可以生成MPS或者LP文件，调用[GLPK](http://www.gnu.org/software/glpk/glpk.html), COIN-OR CLP/[CBC](https://github.com/coin-or/Cbc), [CPLEX](http://www.cplex.com/), [GUROBI](http://www.gurobi.com/), [MOSEK](https://www.mosek.com/), [XPRESS](https://www.fico.com/es/products/fico-xpress-solver), [CHOCO](https://choco-solver.org/), [MIPCL](http://mipcl-cpp.appspot.com/), [SCIP](https://www.scipopt.org/)等求解器来解决线性问题。

# Python环境

为了使用PuLP，首先需要安装一个Python，然后要在你的Python中安装PuLP，到这就完成了Python环境配置了

```shell
pip install pulp
```

————如果你只需要做这一项工作的话。

但是很多时候，我们需要在一台电脑上进行很多工作，而不同的工作需要用到不同的Python包。例如A任务需要a,b,c 三个Python包，而B任务需要d,e,f，随着工作的增多，你会安装越来越多的Python包，Python环境也会越来越臃肿。经过一段时间，当你再次开始A任务时，你发现你的Python环境里面已经安装了上百个软件包，但你只需要用到里面的三个。你开始觉得有些不对劲，不过尽管现在用不到其他软件包但是以后还是会用到，目前也没出什么状况只占用一点硬盘空间似乎并不是什么大问题。

直到有一天，你又需要安装一个新的x包，你发现这次你安装不了了，你遇到了兼容性问题。状况可能是这样，x需要安装a作为依赖，现有环境中已经安装了a，是作为u的依赖安装的，但是很不幸x要求a$\lt 1.0$，而u要求a$\geq 1.0$，因为a在1.0版本发布了重大更新修改了几个重要的API导致无法兼容旧版。你不能卸载掉u然后更新a，因为u还要在其他项目用到；你又不想找个替代u的方法，因为那还要修改你的项目代码还要重新测试；你倒是想找个替代x的方法，但是没找到。

不过经过搜索，你发现很多人遇到了和你同样的问题，终于你在StackOverflow上面还是找到解决方法。尽管x项目已经数年没人维护，不过还是有好心人在GitHub创建了分支将他更新到了兼容a$\geq 1.0$，你下载了好心人的代码按照说明安装完成了x，总算解决了一次兼容性问题。

直到又有一天，你又需要引入一个新的包，这次你没找到stackOverflow的好心人，你开始对着兼容报错发呆；你完成了一个项目想要发布到服务器上，你突然发现你忘了记录项目用了哪些包，你开始对着几十个项目代码文件发呆。

所以在开始怀疑人生之前，为什么不在一开始就把所有项目的Python环境做好隔离呢？

首先，需要一个Python环境的管理工具，常见的有[pipenv](https://github.com/pypa/pipenv), [virtualenv](https://virtualenv.pypa.io/), [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html)，这里我们使用Conda。

## 安装Conda

1. 在[清华镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/?C=M&O=D)下载最新版本的miniconda安装包，按照时间倒序，下载对应操作系统的安装包即可
2. 直接按照默认安装miniconda即可

## 配置Conda和Pip镜像

Conda除了能管理Python环境之外，也可以管理Python包，Pip也一个python包管理软件。Conda和Pip是我们安装、删除Python包用到的主要工具。

Conda和Pip安装软件时会默认从国外的网址（[anaconda.org](), [pypi.org]()）下载代码或二进制文件，为了能在国内能够有一个较好的网络下载质量，首先需要配置一下下载镜像，将下载地址定向到国内的镜像服务器。

创建或者修改`C:\Users\<yourname>\AppData\Roaming\pip\pip.ini`文件，为了避免奇怪的[bom格式](https://www.cnblogs.com/lfire/archive/2012/11/20/2778939.html)请不要使用Windows自带的记事本创建该文件

```ini
[global]
timeout = 120
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

如果你创建文件遇到困难，请尝试在程序目录找到`Anaconda Prompt(MiniConda)`，打开命令行

![image-20201225150318973](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201225154005850.png)

输入

```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.timeout 120
```



创建或者修改`C:\Users\<yourname>\.condarc`，同样不要用记事本。请注意完整的文件名是**`.condarc`**，已经包含了扩展文件名，文件名开头是**`.`**。

如果你创建文件遇到困难，同样你也可以在Anaconda prompt中然后输入`conda config --add channels conda-canary`，也可以生成一个的`.condarc`文件。

将文件内容修改如下

```yaml
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

##  创建第一个Conda环境

1. 打开在Windows程序菜单，打开`Anaconda Prompt(MiniConda)`

   命令行窗口大致如下，最右侧`(base)`表示当前的Conda环境名称，默认的环境名称为`base`，该环境不可被删除。

   ![image-20201225152355916](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201228191557389.png)

   

2. 创建一个Conda环境

   ```shell
   conda create -n pulp python=3.7
   ```

   `-n pulp`表示环境名称，`python=3.7`表示安装python版本。

   可以不指定版本直接使用`python`参数安装与base相同的python版本，也可不添加参数`conda create -n pulp`创建一个完全空的环境。

   在创建环境时还可以同时安装其他软件包，方式与python一样，例如`conda create -n tensorflow python=3.7 tensorflow=2.0`

3. 激活Conda环境

   ```shell
   conda activate pulp
   ```

   激活环境之后，命令行最右侧的名称也会相应改变，变成了`(pulp)`

   ![image-20201225154005850](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201225152355916.png)

4. 安装python包

   激活环境之后，就可以在该python环境下安装项目所需要的软件包。

   如果在第2步没有安装python的话，首先需要安装python

   ```shell
   conda install python=3.7
   ```

   接着使用pip安装pulp，你也可以使用`conda install pulp`

   ```shell
   pip install pulp
   ```

5. 查看环境内安装的包列表

   ```shell
   conda list
   ```

   ```
   (pulp) C:\Users\liukai>conda list
   # packages in environment at C:\Personal\MiniConda\envs\pulp:
   #
   # Name                    Version                   Build  Channel
   amply                     0.1.4                    pypi_0    pypi
   ca-certificates           2020.12.8            haa95532_0    defaults
   certifi                   2020.12.5        py37haa95532_0    defaults
   docutils                  0.16                     pypi_0    pypi
   openssl                   1.1.1i               h2bbff1b_0    defaults
   pip                       20.3.3           py37haa95532_0    defaults
   pulp                      2.4                      pypi_0    pypi
   pyparsing                 2.4.7                    pypi_0    pypi
   python                    3.7.9                h60c2a47_0    defaults
   setuptools                51.0.0           py37haa95532_2    defaults
   sqlite                    3.33.0               h2a8f88b_0    defaults
   vc                        14.2                 h21ff451_1    defaults
   vs2015_runtime            14.27.29016          h5e58377_2    defaults
   wheel                     0.36.2             pyhd3eb1b0_0    defaults
   wincertstore              0.2                      py37_0    defaults
   zlib                      1.2.11               h62dcd97_4    defaults
   ```

   其中最右侧的Channel中pypi的包是由pip安装和管理。

6. 注销Conda环境

   ```shell
   conda deactivate
   ```

   注销之后将回到base环境中

## 其他常用命令

### 查看Conda的信息

```
conda info
```

可以查看conda版本，默认的python版本，安装路径，配置文件路径，镜像服务器（channel urls）地址等信息

```
     active environment : base
    active env location : C:\Personal\MiniConda
            shell level : 1
       user config file : C:\Users\<user>\.condarc
 populated config files : C:\Users\<user>\.condarc
          conda version : 4.9.2
    conda-build version : not installed
         python version : 3.7.6.final.0
       virtual packages : __cuda=10.2=0
                          __win=0=0
                          __archspec=1=x86_64
       base environment : C:\Personal\MiniConda  (writable)
           channel URLs : https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/win-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/win-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/win-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro/win-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro/noarch
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/win-64
                          https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/noarch
          package cache : C:\Personal\MiniConda\pkgs
                          C:\Users\<user>\.conda\pkgs
                          C:\Users\<user>\AppData\Local\conda\conda\pkgs
       envs directories : C:\Personal\MiniConda\envs
                          C:\Users\<user>\.conda\envs
                          C:\Users\<user>\AppData\Local\conda\conda\envs
               platform : win-64
             user-agent : conda/4.9.2 requests/2.25.0 CPython/3.7.6 Windows/10 Windows/10.0.19041
          administrator : False
             netrc file : None
           offline mode : False
```

### 查看Conda环境

```shell
conda env list
```

查看当前已经创建了哪些环境，结果大致如下

```
(base) PS C:\Personal\project> conda env list
# conda environments:
#
base                  *  C:\Personal\MiniConda
ethylene_scheduling      C:\Personal\MiniConda\envs\ethylene_scheduling
flask_by_example         C:\Personal\MiniConda\envs\flask_by_example
flask_service            C:\Personal\MiniConda\envs\flask_service
fpdf                     C:\Personal\MiniConda\envs\fpdf
ipd_api                  C:\Personal\MiniConda\envs\ipd_api
jupyter                  C:\Personal\MiniConda\envs\jupyter
numba                    C:\Personal\MiniConda\envs\numba
or-aps                   C:\Personal\MiniConda\envs\or-aps
or-tools                 C:\Personal\MiniConda\envs\or-tools
pulp                     C:\Personal\MiniConda\envs\pulp
pyomo                    C:\Personal\MiniConda\envs\pyomo
sympy                    C:\Personal\MiniConda\envs\sympy
tensorflow               C:\Personal\MiniConda\envs\tensorflow
test_proj1               C:\Personal\MiniConda\envs\test_proj1
```

分别是环境的名称和路径，默认路径会在miniconda安装目录中的envs文件夹中

### 删除Conda环境

```shell
conda env remove -n <envname>
```

其中`<envname>`为环境名称

也可以直接删除[环境](#### 查看Conda环境)所在的路径

### 导出Conda环境

```shll
conda env export -n pulp > pulp.yml
```

将名称为`pulp`的环境导出到`pulp.yml`的文件中，文件路径在当前命令行所在路径

### 导入Conda环境

```shll
conda create -f pulp.yml
```

### 导出Pip环境

```shell
pip freeze > requirments.txt
```

### 导入Pip环境

```shell
pip install -r requirments.txt
```

## Tips

### 配置Windows Terminal

[Windows Terminal](https://www.microsoft.com/en-us/p/windows-terminal/9n0dx20hk701)是微软开发的一款新式、快速、高效、强大且高效的终端应用程序，适用于命令行工具和命令提示符，PowerShell和 WSL 等 Shell 用户。您可以使用Windows Terminal来替代Anaconda Prompt来获得更好的用户体验。

在Windows Terminal选择设置，打开`settings.json`，在`profiles\list`中添加如下内容

```json
{
	"commandline": "powershell.exe -ExecutionPolicy ByPass -NoExit -Command \"& 'C:\\Personal\\MiniConda\\shell\\condabin\\conda-hook.ps1' ; conda activate 'C:\\Personal\\MiniConda' ; cd 'C:\\Personal\\project' \" ",
	"name": "Conda",
	//"colorScheme": "Night Owlish Light",
	//"icon": "file:///C:\\OneDrive\\File\\icons\\anaconda.png",
	"hidden": false,
    //"cursorColor": "#7a8181"
}
```

`commandline`中有三个路径，分别是`conda-hook.ps1`所在路径，conda安装路径和启动开始的目录

`colorScheme`可以指定Terminal的颜色主题，主题在`settings.json`如下，您可以按需更改

```json
"schemes": [
        {
            "name": "Night Owlish Light",
            "black": "#011627",
            "red": "#d3423e",
            "green": "#2aa298",
            "yellow": "#daaa01",
            "blue": "#4876d6",
            "purple": "#403f53",
            "cyan": "#08916a",
            "white": "#7a8181",
            "brightBlack": "#7a8181",
            "brightRed": "#f76e6e",
            "brightGreen": "#49d0c5",
            "brightYellow": "#dac26b",
            "brightBlue": "#5ca7e4",
            "brightPurple": "#697098",
            "brightCyan": "#00c990",
            "brightWhite": "#989fb1",
            "background": "#ffffff",
            "foreground": "#403f53"
        },
]
```

`icon`可以指定图标路径

`cursorColor`可以指定输入指示标的颜色

# IDE简单使用

人生苦短，既然写python了，当然选择最简单易用的PyCharm作为IDE。

你可以在JetBrains下载到免费社区版本的[PyCharm Community](https://www.jetbrains.com/pycharm/download/#section=windows)，然后安装即可。

1. 打开pycharm，新建一个python项目

2. 选择项目路径

3. 选择已有建立好的conda环境

   ![image-20201228185113826](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201225150318973.png)

4. 完成项目新建

5. 新建一个python脚本`AmericanSteelProblem.py`

   ![image-20201228191557389](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201228191740243.png)

6. 在脚本中输入内容

   ```
   """
   The American Steel Problem for the PuLP Modeller
   Authors: Antony Phillips, Dr Stuart Mitchell  2007
   """
   
   # Import PuLP modeller functions
   from pulp import *
   
   # List of all the nodes
   Nodes = ["Youngstown",
            "Pittsburgh",
            "Cincinatti",
            "Kansas City",
            "Chicago",
            "Albany",
            "Houston",
            "Tempe",
            "Gary"]
   
   nodeData = {# NODE        Supply Demand
            "Youngstown":    [10000,0],
            "Pittsburgh":    [15000,0],
            "Cincinatti":    [0,0],
            "Kansas City":   [0,0],
            "Chicago":       [0,0],
            "Albany":        [0,3000],
            "Houston":       [0,7000],
            "Tempe":         [0,4000],
            "Gary":          [0,6000]}
   
   # List of all the arcs
   Arcs = [("Youngstown","Albany"),
           ("Youngstown","Cincinatti"),
           ("Youngstown","Kansas City"),
           ("Youngstown","Chicago"),
           ("Pittsburgh","Cincinatti"),
           ("Pittsburgh","Kansas City"),
           ("Pittsburgh","Chicago"),
           ("Pittsburgh","Gary"),
           ("Cincinatti","Albany"),
           ("Cincinatti","Houston"),
           ("Kansas City","Houston"),
           ("Kansas City","Tempe"),
           ("Chicago","Tempe"),
           ("Chicago","Gary")]
   
   arcData = { #      ARC                Cost Min Max
           ("Youngstown","Albany"):      [0.5,0,1000],
           ("Youngstown","Cincinatti"):  [0.35,0,3000],
           ("Youngstown","Kansas City"): [0.45,1000,5000],
           ("Youngstown","Chicago"):     [0.375,0,5000],
           ("Pittsburgh","Cincinatti"):  [0.35,0,2000],
           ("Pittsburgh","Kansas City"): [0.45,2000,3000],
           ("Pittsburgh","Chicago"):     [0.4,0,4000],
           ("Pittsburgh","Gary"):        [0.45,0,2000],
           ("Cincinatti","Albany"):      [0.35,1000,5000],
           ("Cincinatti","Houston"):     [0.55,0,6000],
           ("Kansas City","Houston"):    [0.375,0,4000],
           ("Kansas City","Tempe"):      [0.65,0,4000],
           ("Chicago","Tempe"):          [0.6,0,2000],
           ("Chicago","Gary"):           [0.12,0,4000]}
   
   # Splits the dictionaries to be more understandable
   (supply, demand) = splitDict(nodeData)
   (costs, mins, maxs) = splitDict(arcData)
   
   # Creates the boundless Variables as Integers
   vars = LpVariable.dicts("Route",Arcs,None,None,LpInteger)
   
   # Creates the upper and lower bounds on the variables
   for a in Arcs:
       vars[a].bounds(mins[a], maxs[a])
   
   # Creates the 'prob' variable to contain the problem data    
   prob = LpProblem("American Steel Problem",LpMinimize)
   
   # Creates the objective function
   prob += lpSum([vars[a]* costs[a] for a in Arcs]), "Total Cost of Transport"
   
   # Creates all problem constraints - this ensures the amount going into each node is at least equal to the amount leaving
   for n in Nodes:
       prob += (supply[n]+ lpSum([vars[(i,j)] for (i,j) in Arcs if j == n]) >=
                demand[n]+ lpSum([vars[(i,j)] for (i,j) in Arcs if i == n])), "Steel Flow Conservation in Node %s"%n
   
   # The problem data is written to an .lp file
   prob.writeLP("AmericanSteelProblem.lp")
   
   # The problem is solved using PuLP's choice of Solver
   prob.solve()
   
   # The status of the solution is printed to the screen
   print("Status:", LpStatus[prob.status])
   
   # Each of the variables is printed with it's resolved optimum value
   for v in prob.variables():
       print(v.name, "=", v.varValue)
   
   # The optimised objective function value is printed to the screen    
   print("Total Cost of Transportation = ", value(prob.objective))
   ```

7. 运行脚本

   ![image-20201228191658101](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201228191658101.png)

8. 查看结果

   ![image-20201228191740243](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20201228185113826.png)

# 使用PuLP

## 问题描述

考虑一个简单的MIP问题

objective:
$$
\text{minimize  }cost = \sum_{t}(p_tx_t+q_ty_t) +\sum_{t<8}s_th+\frac{s_8h}{2}\\
	t\in\{1,2,\cdots,8\}
$$
subject to:
$$
s_{t-1} + x_t = s_t + d_t \quad 	 t\in\{1,2,\cdots,8\}
$$

$$
x_t \leq (\sum_{k\ge t}^{8}d_k)\times y_t
\quad 	 t\in\{1,2,\cdots,8\}\\
$$

$$
x_t, s_t\in \mathbb{R}_+, \quad y_t\in {0, 1}
$$

其中，参数如下

```python
d = [400, 400, 800, 800, 1200, 1200, 1200, 1200, 800, 800, 400, 400]
h = 5
s_0 = 200
p = 100
q = 5000
```

## 创建问题并选择求解器

```python
import pulp as pl
# create model and solver
model = pl.LpProblem("1-1tiny-problem", pl.LpMinimize)
solver = pl.PULP_CBC_CMD()
```

## 声明对象集合

在该问题中对象$ t\in\{1,2,\cdots,8\}$，所以创建`T`表示，准确来说这是Python里面的一个列表对象，简单地说，就当它数组。

```python
t_num = 8 # t 的数量
T = [0, 1, 2, 3, 4, 5, 6, 7] # t的集合
```

列表T可以通过下标访问其中的元素，下标从0开始。如`T[1]`的值是2，`T[7]`的值为7

## 声明参数的值

```python
q = 5000 # product fix cost
p = 100 # cost per product
d = [400, 400, 800, 800, 1200, 1200, 1200, 1200] # demand of product by month
s_init = 200 # initial stock
h = 200 # stock cost per product month
```

## 声明变量

```python
# variables
x = pl.LpVariable.dicts('x', (T,), 0)
s = pl.LpVariable.dicts('s', (T,), 0)
y = pl.LpVariable.dicts('y', (T,), cat='Binary')
```

对于单个变量，可以直接声明

```python
x = pl.LpVariable('x', 0, 3) # 声明一个[0，3]的变量，名称为x
y = pl.LpVariable('y', cat="Binary") # 声明一个{0， 1}变量y
```

对于一组变量，需要使用`pl.LpVariable.dicts()`声明

```python
x = pl.LpVariable.dicts('x', ([1,2],), 0) # 变量x1,x2，且x>=0
z = pl.LpVariable.dicts('z', ([1,2],[1,2]), cat="Binary") # {0, 1}变量 x11, x12, x21, x22
```

## 添加优化目标

```python
# objective
model += pl.lpSum(p*x[t] + q*y[t] for t in T) + pl.lpSum(h*s[t] for t in T) + h*s[t_num-1]/2
```

`model += `添加约束，但添加的第一条语句会被认为是目标函数

在这里，`pl.lpSum()`函数就是累加函数

## 添加约束

```python
# constraints
# init stock
model += s_init + x[0] == d[0] + s[0]
# first month max limit of x
model += x[0] <= pl.lpSum(d[k] for k in T)*y[0]
for t in T[1:]:
    # stock balance
    model += s[t-1] + x[t] == d[t] + s[t]
    # max limit of x
    model += x[t] <= pl.lpSum(d[k] for k in range(t, t_num))*y[t]
```

`model += `添加约束，右侧则是约束的具体形式

`for t in T[1:]:`表示对T从第2个元素开始遍历 到结尾

## 求解和打印结果

详情查阅基本的python语法

```python
result = model.solve(solver)
for t in T:
    print(f"{x[t]}：{pl.value(x[t])}")

for t in T:
    print(f"{y[t]}：{pl.value(y[t])}")

for t in T:
    print(f"{s[t]}：{pl.value(s[t])}")
```

全部代码如下

```python
import pulp as pl
# create model and solver
model = pl.LpProblem("1-1tiny-problem", pl.LpMinimize)
solver = pl.PULP_CBC_CMD()

t_num = 8 # t 的数量
T = [0, 1, 2, 3, 4, 5, 6, 7] # t的集合


# parameters
q = 5000 # product fix cost
p = 100 # cost per product
d = [400, 400, 800, 800, 1200, 1200, 1200, 1200] # demand of product by month
s_init = 200 # initial stock
h = 200 # stock cost per product month

# variables
x = pl.LpVariable.dicts('x', (T,), 0)
s = pl.LpVariable.dicts('s', (T,), 0)
y = pl.LpVariable.dicts('y', (T,), cat='Binary')

# objective
model += pl.lpSum(p*x[t] + q*y[t] for t in T) + pl.lpSum(h*s[t] for t in T) + h*s[t_num-1]/2

# constraints
# init stock
model += s_init + x[0] == d[0] + s[0]
# first month max limit of x
model += x[0] <= pl.lpSum(d[k] for k in T)*y[0]
for t in T[1:]:
    # stock balance
    model += s[t-1] + x[t] == d[t] + s[t]
    # max limit of x
    model += x[t] <= pl.lpSum(d[k] for k in range(t, t_num))*y[t]

result = model.solve(solver)
# print result
for t in T:
    print(f"{x[t]}：{pl.value(x[t])}")
for t in T:
    print(f"{y[t]}：{pl.value(y[t])}")
for t in T:
    print(f"{s[t]}：{pl.value(s[t])}")
```





