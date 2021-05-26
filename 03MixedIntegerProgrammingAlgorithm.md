# 3 混合整数规划算法

- 3.1 混合整数规划的定义
- 3.2 一种直观的算法运行时间分析方法
- 3.3 分支定界法(Branch and Bound)的形式化
- 3.4 一种先验重构方法来收紧初始数学约束的主要步骤，包括有效不等式、公式的好或坏、理想凸包、严格重构等重要概念
- 3.5 以分离算法(Separation)和切割平面算法（Cut-plane）为基础的分支切割算法
- 3.6 基础的构造和改进启发算法来获得更快更好的可行解，并与分支定界法结合。

## 3.1 混合整数线性规划

**定义3.1** 混合整数规划 MIP
$$
Z(X) = \text{min} \{cx+fy:(x,y)\in X\},
$$

$$
X={(x,y)\in\mathbb{R}^n_+\times\mathbb{Z}^p_+:Ax+By\geq b}
$$

式中：

- $Z(X)$优化目标
- $x,y$表示n为的非负连续变量和非负整数变量
- $c\in\mathbb{R}^n, f\in\mathbb{R}^p$表示优化目标系数
- $b\in\mathbb{R}^m$，为线性约束的边界值
- $A,B$表示线性约束系数

**定义3.2 **0-1整数规划 MBP
$$
Z(X) = \text{min} \{cx+fy:(x,y)\in X\}
$$

$$
X={(x,y)\in\mathbb{R}^n_+\times\{0,1\}^p:Ax+By\geq b}
$$

0-1整数规划的约束可以通过线性松弛转换为线性约束，我们定义$P_X$为线性松弛问题的定义域
$$
P_X = \{x, y\}\in\mathbb{R}_+^n\times[0,1]^p:Ax+By\geq b
$$
**定义3.3** 线性松弛的整数规划
$$
Z(P_X) = \text{min} \{cx+fy:(x,y)\in X\}
$$

$$
P_X = \{x, y\}\in\mathbb{R}_+^n\times\mathbb{R}_+^p:Ax+By\geq b, P_X\supseteq X
$$

式中：

- $P_X$为松弛后的的解定义域

**推论3.1**对于最小优化目标来说，松弛后的最小目标值小于或等于原值，为目标的下限。
$$
Z(P_X)\leq Z(X)
$$
**推论3.2**对于最小优化目标来说，任意的可行解都是优化目标的上限
$$
Z(X)\leq \bar{Z}
$$




下面通过一个简单的案例，来演示MIP问题中如何使用分支定界和剪枝算法。该问题中只有两个整型变量，是一个纯整数规划问题（pure integer programming)
$$
Z(X) = \text{min} \{-y_1-2y_2:y=(y_1,y_2)\in X\}
$$

$$
\begin{aligned}
X=\{y=(y_1, y_2)\in \mathbb{Z}_+^2:& y_1&\geq& 1\\
&-y_1 &\geq& -5\\
&-y_1-0.8y_2 &\geq&-5.8\\
&y_1-0.8y_2 &\geq&0.2 \\
&-y_1-8y_2 &\geq&-26
\}\end{aligned}
$$

在该问题中，$n=A=c=0$
$$
m=5,p=2,f=(1,2),B=\begin{pmatrix}&1 &0\\
&-1&0\\
&-1 &-0.8\\
&1&-0.8\\
&-1&-8\end{pmatrix}
,b=\begin{pmatrix}
-1\\
-5\\
-5.8\\
0.2\\
-26
\end{pmatrix}
$$
![image-20210104165237403](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20210104165237403.png)

该问题的可行解，松弛后的约束范围以及目标方向可以用图3.1表示

## 3.2 算法运行时间

在开始求解MIP问题之前，首先了解以下一些算法评价和分析的标准方法。

### 3.2.1 算法的性能

考虑有问题$P$和算法$A$

评价算法有两个维度，速度和质量。速度的时值算法的运行时间，质量则是算法得到目标值的好坏（例如 duality gap）。算法的运行时间通常并直接使用运行时间来评价，而是使用其[复杂度](https://zh.wikipedia.org/wiki/%E6%97%B6%E9%97%B4%E5%A4%8D%E6%9D%82%E5%BA%A6)来评价，该指标于硬件速度、软件平台、编译器等外界因素无关。

问题可以按照其数学模型的结构进行分类，对于同一类问题来说，算法求解的耗时有一定共性。例如对于LS-U的问题，其复杂度与维度有关，为
$$
N(A,n) = 3\frac{n(n-1)}{2} + n^2
$$
更确切的，我们使用$O(n^p)$来表示复杂度与问题规模的关系，$O(n)$表示随问题规模线性增长，$O(n^2)$表示随问题规模的平方增长。

**问题的规模**

对于LS-U问题，通常由$3n$个变量和$2n$条约束，仪表我们使用
$$
O(n)\times O(n)
$$
来表示其规模，规模并不表示其复杂度，只是表示其问题大小。

## 3.3 分支定界法

回顾以下之前的PIP问题

我们首先将MIP问题通过线性松弛变换成了如下形式
$$
Z(V) = \text{min}\{cx+fy:(x,y)\in V\}
$$
式中$V$表示问题的线性空间

问题的原始约束如下
$$
X = \{(x,y)\in\mathbb{R}_+^n\times \mathbb{Z}_+^p:Ax+By\geq b\}
$$
线性松弛后的最优解$Z(P_X)$的线性空间如下
$$
P_X = \{(x,y)\in\mathbb{R}_+^n\times\mathbb{R}_+^p:Ax+By\geq b\}
$$

问题的上限和下限
$$
Z(P_X)\leq Z(X)\leq \bar{Z}
$$
式中$\bar{Z}$表示问题任意的可行解

### 3.3.1 枚举原则

首先说明分支定界法在解决MIP问题时，最基本的分治原则。

1. $Z(X)$的下限来自于线性松弛后的最优解$Z(P_X)$。线性问题通常可以快速求解，令$(x^*,y^*)$为线性问题的最优解。

   **假设**  一般来说，我们假设线性松弛后的问题是有界的，否则$Z(P_X)=-\infty$，则MIP问题将是无边界或者是不可行。为了区分这两种情况，我们给变量设置一个足够大的上下限，然后运行分支定界法，如果找到了最优解，那么问题无边界否则问题不可行。

   求解MIP问题，可以理解在线性空间$P_X$ 内寻找最优的$(x,y)$且满足$y\in\mathbb{Z}^p$

   **原则**  我们通过求解一系列LP问题来尝试求解MIP问题

2. 如果$y^*\in\mathbb{Z}^p$，那么$(x^*,y^*)\in X$也满足，所以问题的上限和下限相等，$(x^*,y^*)$为最优解。
$$
  cx^*+fy^*=Z(P_X)\leq Z(X)\leq \bar{Z}=cx^*+fy^*
$$


3. 如果$y^*\notin\mathbb{Z}^p$，那么$(x^*,y^*)$不是MIP的可行解。我们会通过添加一些线性约束的方式，来消除无用的线性松弛。

   令$y_j,j\in\{1,2,\cdots,p\}$表示线性松弛问题最优解$y^*$中非整数的变量。

   **注意**  对于任何的可行解$(x,y)\in X$，显然有$y_j\le\lfloor y^*_j\rfloor$或者$y_j\geq \lceil y^*_j\rceil$，$\lfloor y_j^*\rfloor,\lceil y^*_j\rceil$分别表示对$y_j^*$向下和向上取整。例如，$y_j^*=\frac{32}{9}$，则有$y_j\leq 3=\lfloor\frac{32}{9}\rfloor$或者$y_j\geq 4=\lceil\frac{32}{9}\rceil$

   **分支**  为了消除$(x^*,y^*)$中的不可行值，我们将$P_X$分成$P_X^0,P_X^1$两个部分
   $$
   P_X^0=P_X\cap\{(x,y)\in\mathbb{R}_+^n\times \mathbb{R}_+^p:y_j\leq \lfloor y_j^*\rfloor\}
   $$

   $$
   P_X^1=P_X\cap\{(x,y)\in\mathbb{R}_+^n\times \mathbb{R}_+^p:y_j\geq \lceil y_j^*\rceil\}
   $$

   式中，$y_j$被称为分支变量，$y_j\leq \lfloor y_j^*\rfloor, y_j\geq\lceil y_j^*\rceil$被称为分支约束。

   现在，我们可以将问题由在$P_X$中寻找最优解替换成在$P_X^0\cup P_X^1$中寻找最优解，代价就是我们将一个线性问题变成了两个线性问题。

   **案例**  图3.1中的问题，经过线性松弛后，得到解$(x^*,y^*)$为$a=(\frac{32}{9},\frac{101}{36})$，将$P_X$在分支变量$y_1$处分成$P_X^0,P_X^1$ ，如图3.2所示

   ![image-20210105105349131](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20210105105349131.png)

4. 我们继续在分解后的线性空间$L=\{P_X^0,P_X^1\}$寻找最优解，在遇到非整数解是继续将问题分解，并添加到列表$L$中，然后不断递归。

   **主迭代** 我们给出列表$L$和当前的最佳可行解$\bar{Z}$，没有可行解时，令$\bar{Z}=+\infty$

   **选择和求解**  选择列表L中的一个线性空间$V$，求解线性规划问题，得到$Z(V)$和最优的变量取值$(x^V,y^V)$，$Z(V)$就是线性空间$V$内的目标下限值

   **剪枝**  在线性空间$V$内可能由以下几种情况

   a. 如果$Z(V)\geq \bar{Z}$，那么空间$V$内不存在严格优于$\bar{Z}$的解，我们不需要在检验$Z(V)$的边界是否在整数上，直接从列表$L$中删除$V$即可，这是修剪边界

   b. 一种特殊情况是如果$V$是空集，我们可以得到$Z(V)=+\infty\geq \bar{Z}$，也可以排除$V$，这是修剪不可行域

   c. 如果$Z(V)<\bar{Z},y^V\in \mathbb{Z}^p$，那么我们在空间$V$可以找到最优解$Z(V)$和对应的变量值$(x^V,y^V)$，且该解优于当前的可行解，因此令$\bar{Z}=Z(V)$，然后将$V$从列表中移除。

   d. 如果$Z(V)<\bar{Z},y^V\notin\mathbb{Z}^p$，那么空间$V$内的解不是整数，因此需要将$V$分支成$V^0,V^1$，然后添加到列表$L$中，这被称为分支。

5. **终止条件** 算法会在列表$L$为空后终止，如果$y$是有限的话，此时算法可以保证在有限步骤内找到全局的最优解。需要注意的是，列表$L$的空间数量会随着整数变量维度$p$指数增长。

   **复杂度** 理论上分支定界法的迭代次数随整数变量的维度$p$指数增长，每次迭代都包括一次线性规划的计算（大约是多项式复杂度）

### 3.3.2 分支定界算法

我们将分支定界算法归纳如下

1. *初始化*

   *$L=\{P_X\}$*

   *$\bar{Z}=+\infty$*

   *假设$Z(P_X)>-\infty$*

2. *终止条件*

   *if $L=\empty$*

   ​	*if $\bar{Z}=+\infty$，问题无解（infeasible）*

   ​	*if $\bar{Z}<+\infty$，最优解为$\bar{Z}$*

   ​	*stop*

3. *节点选择和求解*

   *选择$V\in L$，令$L=L\backslash \{V\}$*

   *对子空间$V$进行线性求解得到$Z(V),(x^V,y^V)$*

4. *剪枝*

   *if $Z(V)\geq \bar{Z}$，goto 2*

   *if $Z(V)<\bar{Z}$，*

   ​	*if $y^V\in\mathbb{Z}^p$*

   ​		*update $\bar{Z}=Z(V)$*

   ​		*update $L$: for $W\in L$ : if $Z(W)\geq \bar{Z}$，$L=L\backslash{W}$*

   ​		*goto 2*	

5. *分支*

   *$L=L\cup\{V^0,V^1\}$*

   *$V^0 = V \cap \{ (x,y) \in \mathbb{R}_+^n \times \mathbb{R}_+^p : y_j \leq \lfloor y_j^V \rfloor \}$*

   *$V^1=V \cap \{ (x,y) \in \mathbb{R}_+^n \times \mathbb{R}_+^p : y_j \geq \lceil y_j^V \rceil\}$*

   *goto 2*

分支列表$L$通常会使用枚举树来表示其结构，下图为案例的分支遍历，以及对应点在问题空间的未知。

![image-20210106142628381](https://gitee.com/behe-moth/picgo_img/raw/master/pic/branch_and_bound_flowchart2.svg)

下面为一个案例问题的分支定界方法的结果。

![image-20210106093324276](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20210106092116750.png)

![image-20210106092116750](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20210106093324276.png)

### 3.3.3 节点选择和分支的策略

节点选择指在子空间列表$L$选择下一个节点进行计算步骤。选择的节点关系到算法搜索过程中的上下限的收敛速度，一般来说常见的策略有：

- DFS，深度优先
- BFS，广度优先
- 最佳边界，选择具有最小的下限的节点
- 混合策略：先BFS后DFS，先DFS后最佳边界等

大多数的选择策略都是启发式的。DFS可以较快的获得一个可行解（通常来说，遍历越深，越容易得到整数解），但是解的质量难以保证。BFS则更容易找到全局较优的分支，但是在寻找可行解时效率会变差。

分支策略是如何选择分支变量的步骤。最常用的是选择$y^*$最接近0.5的变量。

大多数的MIP求解器都会包含基础的选择和分支策略，也支持实现自定义的策略。

### 3.3.4 结果的质量

优化结果的好坏通常使用[对偶间隙](https://zh.wikipedia.org/wiki/%E5%B0%8D%E5%81%B6%E9%96%93%E9%9A%99)（Duality Gap）来评价。
$$
Duality Gap = \frac{Best UB-Best LB}{BestLB}\times100\%
$$
$BestUB$为最佳的可行解，$BestLB$为最佳的线性松弛解。需要注意的是，大多数优化求解器使用$BestLB$作为对偶间隙的分母。

## 3.4 重构

使用分支定界法就可以完成MIP问题的求解，但是难点在于，大多数实际的问题，变量太多，导致计算复杂度指数增长，以至于在有限时间内无法保证结果。因此多数时候，需要对模型的方程进行适当的重构，改善模型的求解性能。

### 3.4.1 方程的质量

**方程的好与坏**

**结论3.3** *分支定界法在求解MIP问题时需要计算的节点数量很大程度上取决于方程形式，因此需要掌握如何辨别方程的好坏。*

在PIP的案例当中，我们在图3.3中的求解共遍历了9个节点，其方程形式如下
$$
\begin{aligned}
X=P_X\cap\mathbb{Z}_+^2\quad \text{and} \quad P_X=\{y=(y_1,y_2)\in\mathbb{R}_+^2\}:&y_1&\geq&1\\\
&-y_1&\geq&-5\\
&-y_1-0.8y_2&\geq&-5.8\\
&y_1-0.8y_2&\geq&0.2\\
&-y_1-8y_2&\geq&-26
\end{aligned}
$$
可行域范围在图3.1中阴影表示，观察图形可以发现，对于所有的可行解来说，可以新增一些约束进一步缩小范围
$$
\begin{aligned}
-y_1-y_2 & \geq&  6, \\
y_1-y_2  & \geq&  0, \\
-y_2      & \geq& -2.
\end{aligned}
$$
![](https://gitee.com/behe-moth/picgo_img/raw/master/pic/image-20210106142628381.png)
在图3.5中表示处新的约束范围后发现，原来的MIP问题只需要一次LP计算就可以得到结果，因为最优解$f$就在约束边界上。



**严格或凸的公式化**

上述例子中的方程形式可以使用纯粹的LP方法来求解PIP问题，这是因为新的线性松弛与$X$的凸包完全一致。这些新增的约束被称为有效不等式。$X$的凸包，$\text{conv}(X)$可以理解为最小的或者最严格的包含所有$X$点的公式化线性约束，同时它也是最佳的公式化约束形式。$X$中所有的边界点都在约束边界，因此无论目标函数如何变化，最终都可以通过LP一次找到最佳解。

**结论3.4** *MIP问题都可以通过一组表示可行域$X$的凸包线性有效不等式将原问题重构，转换成一个纯粹的LP问题求解。这种方式，也被称为严格公式化。用来描述$X$的凸包的不等约束又被称为多面体。*

但是，在实际的应用中，该方法有两个主要难点。

1. $X$的凸包是未知的，寻找凸包通常来说是和求解MIP问题同样复杂的。
2. 大多数MIP问题的$\text{conv}(X)$的方程个数随变量数同样有指数增长的问题，这可能导致线性问题难以求解。

因此，我们不能指望在实际的应用中使用你先的严格公式化方法。我们会在3.5继续讨论第二点困难。

### 3.4.2 有效不等式

在开始优化之前，添加一些有效不等式或者约束有以下目的：

- 获得一个严格的公式化形式
- 提高下限值
- 减少求解过程中分支定界法的节点数量
- 减少分支定界法的耗时

**定义3.10** 有效不等式(Valid Inequalities, VI)定义如下：
$$
\forall\quad (x^{\star},y^{\star})\in X, \quad \alpha x^\star+\beta y^\star\geq \gamma\\
\begin{aligned}
\text{with}\quad& X=\{(x,y)\in\mathbb{R}_+^n\times\mathbb{Z}_+^p:Ax+By\geq b\},\\
& \alpha \in \mathbb{R}^n, \beta\in\mathbb{R}^p,\gamma\in\mathbb{R}
\end{aligned}
$$
换句话说，$\alpha x^\star+\beta y^\star\geq \gamma$是有效不等式的充分条件是该不等式需包含$X$内部所有的点。

显然并不是所有的有效不等式都是我们关心的，我们关心的VI更多是那些严格的处于边界的不等式，只有这些不等式才有利于改善分支定界法的搜索性能。

**定义3.11**  *$X$的面指描述$\text{conv}(X)$多面体的必要的不等式。*

通常来说，定义多面体的有效不等式是非常困难的，因此，更多时候我们只能对模型进行部分的收紧，以期望可以减少分支数量和算法耗时。

**松弛分析和有效不等式寻找**

在数学上有一些判断、分类可行域$X$的有效不等式的方法，但是这些方法的解释已经超过本书的范畴。这里我们简单的演示一些通用的方法来帮助读者足够使用已有的重构结论和软件，更好的理解重构和将为方法，并为第二部分做准备。

通常来说，判断凸包$X$的面需要全面考虑其结构，显然这过于困难，因此我们一般会使用其超集$Y$来简化问题。

**结论3.5** *超集$Y$的任何有效不等式也是$X$的有效不等式*

$Y$的一种低层面的松弛方式是考虑$X$中一条不等式，然后对变量的上下限进行约束。

回顾案例的PIP问题
$$
\begin{aligned}
X=\{y=(y_1, y_2)\in \mathbb{Z}_+^2:& y_1&\geq& 1\\
&-y_1 &\geq& -5\\
&-y_1-0.8y_2 &\geq&-5.8\\
&y_1-0.8y_2 &\geq&0.2 \\
&-y_1-8y_2 &\geq&-26
\}\end{aligned}
$$
首先对变量$y_1$有显示的上下限约束$1\leq y_1\leq 5$

对于$y_2$， 因为$26\geq y_1+8y_2\geq8y_2$显然有$0\leq y_2\leq 3$

考察约束$y_1+0.8y_2\leq 5.8\Rightarrow y_1+y_2\leq 5.8+0.2y_2\leq 6.4 \Rightarrow y_1+y_2\leq 6$

这样我们得到了$X$的一个超集$Y$
$$
\begin{aligned}
Y=\{y=(y_1,y_2)\in\mathbb{Z}_+^2:1 &\leq y_1 \leq5\\
&0\leq y_2\leq 3\\
&y_1+y_2\leq 6\}.
\end{aligned}
$$
同样，我们还能得到另一条有效不等式$y_1-0.8y_2\geq0.2\Rightarrow y_1-y_2\geq0.2-0.2y_2\geq-0.4\Rightarrow y_1-y_2\geq0$

大多数先进的MIP求解器可以在除了约束系数矩阵之外，对模型结构没有任何先验知识的情况下，自动得出这种低层面松弛的有效不等式。 这种不等式的例子包括背包问题，混合整数舍入（MIR）和我们在第二部分中研究的Flow Cover VI。 它们全部基于单个约束和变量范围。

我们更感兴趣的是在某些情况下如何获得更全面高等级的松弛$Y$，从而更好的重构$\text{conv}(X)$。我们会在第4章讨论典型的LS-U问题的有效不等式和松弛的重构。

### 3.4.3 先验重构

在我们进行了一系列松弛分析之后，我们获得了一组有效不等式$C$
$$
\mathcal{C}=\{\alpha^jx+\beta^j\geq\gamma^j\quad \text{for all } j=1,\cdots,|\mathcal{C}|\}
$$
回顾原问题线性松弛后的空间$P_X$
$$
P_X=\{(x,y)\in\mathbb{R}_+^n\times\mathbb{R}_+^p:Ax+By\geq b\}
$$
原问题可行解的空间$X$
$$
X=P_X\cap(\mathbb{R}_+^n\times\mathbb{Z}_+^p)
$$
**定义3.12**  *使用$\mathcal{C}$对$X$重构，相当于把$\mathcal{C}$的约束添加到X中*
$$
\begin{aligned}
X = \{ (x,y)\in\mathbb{R}_+^n\times\mathbb{Z}_+^p:&Ax+By\geq b \} \\
  = \{ (x,y)\in\mathbb{R}_+^n\times\mathbb{Z}_+^p:&Ax+By\geq b \\
  &\alpha^jx+\beta^j\geq\gamma^j\quad \text{for all } j=1,\cdots,|\mathcal{C}|
  \}
\end{aligned}
$$
*重构后的$X$仍然会包含原来的可行解空间，但是会得到一个更严格的线性松弛空间$\tilde{P}_X$*
$$
\tilde{P}_X=P_X\cap C \subseteq P_X
$$
*$C$为有效不等式的约束空间*
$$
C = \{(x,y)\in \mathbb{R}_+^n\times\mathbb{R}_+^p:\alpha^jx+\beta^jy\geq\gamma^j\quad \text{for all } j=1,\cdots,|\mathcal{C}|\}
$$
以上就是先验重构的方法，在实际应用过程中，先验重构一般会在$|\mathcal{C}|$不是非常大的时候使用。

**先验重构案例**

对于$P_X$
$$
\begin{aligned}P_X=\{y=(y_1, y_2)\in \mathbb{R}_+^2:& y_1&\geq& 1\\&-y_1 &\geq& -5\\&-y_1-0.8y_2 &\geq&-5.8\\&y_1-0.8y_2 &\geq&0.2 \\&-y_1-8y_2 &\geq&-26\}\end{aligned}
$$
使用$C$进行先验重构
$$
\begin{aligned}
C=\{(x,y)\in\mathbb{R}_+^n\times\mathbb{R}_+^p:&y_1+y_2\leq6\\
&y_1-y_2\geq 0 \}
\end{aligned}
$$
重构后的$\tilde{P}_X$为图3.7中的阴影部分，使用分支定界法对$\tilde{P}_X$进行求解，可以看到相比于未重构的9个节点，重构后仅需5个节点。

![image-20210107172753736](https://gitee.com/behe-moth/picgo_img/raw/master/pic/20210113141804.png)

![image-20210111163851179](https://gitee.com/behe-moth/picgo_img/raw/master/pic/20210202124802.png)

### 3.4.4 扩展重构

**定义3.13**  *引入新的变量的重构方式被称为扩展重构（在第6章详细介绍）。*

通过在新的变量空间添加约束，扩展重构能够使用远少于原始空间的约束来重构凸包，但是会带来变量个数的增加。

通常，对于一个$O(n)$的问题，扩展重构需要增加$O(n^2)\sim O(n^3)$个扩展变量以及$O(n^2)\sim O(n^3)$个约束；相较于原始问题的$O(2^n)$，可以大大减少约束数量和计算复杂度。扩展重构的案例可以参见第4章的LS-U问题。

另外，扩展重构的方程可以直接被添加到原方程之中，而对原空间的指数增长复杂度问题的重构还需要借助3.5节的切割平面法和分支切割算法。

## 3.5 分支切割算法

使用有效不等式进行重构时，另一个困难之处在于指数增长的复杂度，尤其是对原问题空间的重构。因此，在实际中不会一次将所有的有效不等式都进行重构，而是在有效的时间内寻找一些较有的重构方式，从而缩紧$\tilde{P}_X$，提高下限，减少节点数量。

**结论3.6**  *对于一个固定的目标方程$cx+fy$的MIP问题，只需要很少（大多数情况下等于$n+p$个）的有效不等式就可以获得一个最优重构的线性松弛空间$\tilde{P}_X$。*

因为线性松弛后的最优解一定在$n+p$维的凸包顶点上，如果将不包含最优解的有效不等式删除后，那么描述一个$n+p$维的最优点需要$n+p$条线性约束。

### 3.5.1 分离算法

结论3.6说明，大多数情况下不需要将所有的有效不等式都添加到模型当中，只需要在必要的时候将有效不等式添加即可大大减小问题的规模，这也是分离算法和切割平面算法对 $\tilde{P}_X$进行改进出发点。

分离算法解决的主要问题是，给定一组约束$\mathcal{C}$和一个点$(x,y)$，判断$(x,y)$是否在$\mathcal{C}$内。

**定义 3.14** 给定的可行域$X$和其有效不等式约束$\mathcal{C}$，对于$(x^\star,y^\star)\in P_X$，分离问题$SEP((x^\star,y^\star)|\mathcal{C})$的定义为：证明$(x^\star,y^\star)\in C$或者找到$\alpha^jx^\star+\beta^jy^\star\ngeq\gamma^j$

求解分离问题的算法即为分离算法，毫无疑问对于指数增长的有效不等式，设计高效的分离算法是非常困难的。分离算法是获得切割平面法复杂程度的重要步骤。分离算法可以分为精确和启发两类，启发式算法常被用于加速计算。

### 3.5.2 切割平面法

我们首先回顾一下之前的内容。

对于MIP问题和其定义域$X$和最小化目标$Z(X)$。

我们通过线性松弛可以获得空间$P_X$，$P_X$是一个线性空间很容易找到最优值$\bar{Z} = Z(P_X)$和解$(x^*,y^*)$，$\bar{Z}$是问题的下限，这个过程也被称为定界。

但是$(x^*,y^*)$在$P_X$空间内，并不能保证满足整数约束。对于$y*\notin\mathbb{Z}_+^p$，需要在变量$y^j\notin\mathbb{Z}_+$上进行分支，将$P_X$分成两个子空间$P_X^0,P_X^1$，并在子空间内不断进行定界和分支操作，直到遍历完所有分支列表$L$内的所有分支。

由于分支会随着变量个数指数增长，为了保证求解效率，受凸包的启发我们希望能尽可能的找到一些有效不等式$\mathcal{C}$，将$P_X$缩小为$\tilde{P}_X$，从而提高分支效率，减少分支数量。

有效不等式可以根据每一条约束以及变量上下限获得，但是有效不等式的数量也会随着变量个数指数增长，尤其是在分支的子空间中，仍然使用所有的有效不等式显然是不合适的。我们希望可以找到尽可能少的只考虑那些对优化目标有影响的有效约束，这些就是分支切割算法。为了完成分支切割，首先需要判断$(x^*,y^*)$是否满足$\mathcal{C}$，也就是分离算法。

就像我们使用有效不等式来获得缩小后的空间$\tilde{P}_X$来改进$P_X$，切割平面法需要解决的问题就是，对于分支列表$L$中的每一个节点$V$，如何利用有效的不等式$\mathcal{C}$，来获得缩小的严格的$\tilde{P}^V_X=V\cap C$。问题在于$\mathcal{C}$是对$P_X$所有变量的限制，对于其子空间$V$来说，显然过于大了，因此需要一些精简。

切割平面法的步骤如下：

1. 令$W:=V$，在没有有效不等式$\mathcal{C}$的情况下计算线性松弛目标$Z(W)$，其解记作$(x^*.y^*)$

2. 计算分离问题$\text{SEP}((x^*,y^*|\mathcal{C}))$

3. 如果$(x^*,y^*)$全部满足$\mathcal{C}$，也就是$(x,y^*)\in V\cap C$，$(x^*,y^*)$为$Z(\tilde{P^V_X})$的解，结束

4. 如果$(x^*,y^*)$不满足$\mathcal{C}$，将不满足的约束$\alpha^jx^*+\beta^jy^*\ngeq\gamma^j$添加到$W$，

   令$W:=W\cap \{(x,y)\in \mathbb{R}_+^n \times \mathbb{R}_+^p:\alpha^jx^*+\beta^jy^*\geq\gamma^j\}$

5. 计算新的$Z(W)$和$(x^*,y^*)$，到第2步。

在该方法中，分离算法返回的结果被称为切割平面，表示对解$(x^*,y^*)$的切割的超平面。根据结论3.6，我们期望可以通过少量分离操作迭代，将有效不等式的数量大幅削减，从而避免了添加了全部有效不等式后指数爆炸的线性问题的复杂度问题。我们也可以在一次迭代中，添加多个不满足的约束，减少分离算法的次数。

### 3.5.3 分支切割算法

综合以上，分支切割算法可以描述如下，在分支定界法的基础上，使用有效不等式$\mathcal{C}$对分支空间$V$进行重构，使用切割平面法对分支空间进行定界。

对于初始节点，通常使用切割分支法（因为初始节点需要先切割可以减少初始节点数量）来收紧约束。

为了能够加速分离算法，通常会有一个*切割池*，缓存一部分有效不等式约束或者切割平面，因为对于父节点成立的超平面大概率对其子节点也是成立的，该方法又被称为*切割池分离*。

分支切割算法的总结如下

1. 初始化

   $L=\{P_X\}$

   $\bar{Z}=+\infty$

   Assume LR is bounded。

2. 终止判断

   If $L=\empty$ Then

   {    If $\bar{Z}=\empty$ Then $X=\empty$, problem infeasible;

   ​      If $\bar{Z}>+\infty$ Then $\bar{Z}$ is optimal;

   ​     Stop.

   }	

3. 节点选择和切割平面法求解

   **SELECT** $V\in L$, **LET** $L:=L\backslash V$

   3.a **COMPUTE** $Z(V)$ and solution $(x^V, y^V)$ of $V$

   3.b **SOLVE** $\text{SEP}((x^V,y^V|\mathcal{C}))$

   ​    **IF** $(x^V,y^V)$ satisfied with $\mathcal{C}$ **THEN** $(x^V, y^V)$ is LP optimal of $Z(V\cap C)$ , **GOTO** 4

   ​	**ELSE** $(x^V,y^V)$ is violated with $\alpha^jx^V+\beta^jy^V\geq\gamma^j$ **THEN**

   ​        {**ADD**  $\alpha^jx^V+\beta^jy^V\geq\gamma^j$ to V, $V:=V\cap\{(x,y)\in\mathbb{R}_+^n\times\mathbb{R}_+^p:\alpha^jx^V+\beta^jy^V\geq\gamma^J\}$

   ​        **COMPUTE** $Z(V)$ and solution $(x^V,y^V)$

   ​        **LET** $(x^V,y^V)$ as LP optimal

   ​        **GOTO** 3.b }

4. 剪枝

   **IF** $Z(V)\geq\bar{Z}(V)$, **THEN GOTO** 2;

   **ELSE** $Z(V)<\bar{Z}(V)$ **THEN**

   ​    **IF** $y^V\in \mathbb{Z}_+^p$ **THEN** 

   ​        {**UPDATE**  $\bar{Z}(V):=Z(V)$

   ​        **UPDATE**  $L:=\{W \quad \text{for} \quad W\in L \quad\text{if}\quad Z(W)<\bar{Z}(V)\}$ 

   ​         **GOTO** 2}

   ​	**ELSE** $y^V_j\notin\mathbb{Z},\quad j\in\{1,2,\cdots,p\}$ **THEN** **GOTO** 5.

5. 分支

   **SELECT** $j$ from $y^V_j\notin Z$

   **LET** $V^0 = V \cap \{(x,y)\in\mathbb{R}_+^n\times\mathbb{R}_+^p:y_j^V\leq \lfloor y_j^V \rfloor\}$,  $V^1 = V \cap \{(x,y)\in\mathbb{R}_+^n\times\mathbb{R}_+^p:y_j^V\geq \lceil y_j^V \rceil\}$

   **UPDATE**   $L:=L\cup \{V^0,V^1\}$
   
   **GOTO** 2
   
   
   

**切割平面和分支切割算法案例**
对于问题3.1 
$$
   Z(X) = \text{min} \{-y_1-2y_2:y=(y_1,y_2)\in X\}
$$

$$
   \begin{aligned}X=\{y=(y_1, y_2)\in \mathbb{Z}_+^2:& y_1&\geq& 1\\&-y_1 &\geq& -5\\&-y_1-0.8y_2 &\geq&-5.8\\&y_1-0.8y_2 &\geq&0.2 \\&-y_1-8y_2 &\geq&-26\}\end{aligned}
$$

有效不等式如下
$$
\mathcal{C} = \bigg\{\begin{aligned}
y_1+y_2\leq 6\\
y_1-y_2\geq 0
\end {aligned}\bigg\}
$$
采用切割平面发进行分支定界算法求解，会得到与先验重构图3.6相同的结果。不同之处在于，先验重构的$Z(V)$会重构所有的有效不等式，而切割平面法则只会重构必要的有效不等式。

![image-20210111163641257](https://gitee.com/behe-moth/picgo_img/raw/master/pic/20210113141804-2.png)

例如图3.8所示，在求解第一个线性最优点$a'$，切割平面法只用到了$y_1+y_2\leq6$这一条有效不等式。

![image-20210107172739361](https://gitee.com/behe-moth/picgo_img/raw/master/pic/20210202124744.png)



根据图3.8，切割平面法的过程如下：

1. 令$W= P_X$，计算得到线性最优解$a$
2. 求解$\text{SEP}(a|\mathcal{C})$，得到不满足的有效不等式$y_1+y_2\leq 6$
3. 将$y_1+y_2\leq 6$添加到$W$中(即图中阴影部分)，求解得到新的线性最优解$a'$
4. 求解$\text{SEP}(a'|\mathcal{C})$，发现$a'$满足$\mathcal{C}$，将$a'$作为线性最优解进行剪枝判断

## 3.6 启发方法————寻找可行解

前面提到的方法都是在改进线性问题来提高混整问题的下限，我们还可以通过启发式方法找到更好的可行解，来改善问题的上限。我们期望找到一种寻找可行解的方法，来加速分支定界法。本节介绍的大部分方法均可以运用到所有节点之上，为了便于表述我们均使用根节点作为案例。对于该节点，还会有一个通过$P_X$或者收紧的$\tilde{P}_X$得到的线性最优解$(\hat{x},\hat{y})$，以及一个可能的当前最优可行解$(\bar{x},\bar{y})$。

通常来说，启发式分为两种，构造启发方法——寻找可行解和改进启发方法——寻找比当$(\bar{x},\bar{y})$更好的解。为了简化问题，我们假设$y\in\{0,1\}^p$

**截断MIP**

截断MIP设定时间限制，然后运行分支定界法或者分支切割法，在时限内找到的解作为启发解。该方法并不适用，尽管可以同时用于两种启发式目的。

**深度遍历**

深度遍历的策略是使用DFS进行分支遍历，然后每次分支将其中一个非整数的$y_k$固定为整数。

LP-深度遍历的策略是每次固定距离整数最近的非整数解。具体来说，对于当前的线性最优解$(\hat{x},\hat{y})$，对于非整数集$F=\{y_j\notin\mathbb{Z}^1\}$，寻找$g_k=\text{min}_{j\in F}g_j$，其中$g_j=\text{min}[\hat{y},1-\hat{y}]$，如果$\hat{y}_k\leq0.5$，令$y_k=0$ ，否则令$y_k=1$。

IP-深度遍历，则会每次将$y_k$固定为当前最优解$\bar{y}_k$。

显然LP-深度遍历适用于寻找可行解，而IP深入遍历则适用于改善最优解。大多数求解器对该方法都有很好的实现，因而我们并不会深入讲解该方法。

接下来我们会介绍几种启发式方法，这些方法通常都是在LP问题或者原始的MIP问题上做了一点改动，例如固定一些变量、增加一些约束或者松弛一些变量等，这些方法通常会使我们更容易找到启发解或者找到更好的启发解。

### 3.6.1 构造启发解

我们介绍两种构造启发解的方法。

首先，令$Q=\{1,2,\cdots,p\}$作为变量$y$的索引。

**线性固定法或者切割固定法**

固定线性最优解中的整数部分，然后求解剩下的$\text{MIP}^{\text{LP-FIX}}$，该问题表述如下
$$
(\text{MIP}^{\text{LP-FIX}})  \text{  min}\{cx+fy:Ax+By\geq b,x\in\mathbb{R}_+^n,y\in\{0,1\}^p,\\
y_j=\hat{y}_j, \text{for all }j\in Q \text{ with } \hat{y}_j \in\{0,1\}\}.
$$


一般来说，该方法在非整数解较少且收紧的线性约束上，可以得到更好的结果。切割固定与之类似，在重构的时候固定那些整数的切割面，而不是选择扩展切割方程。

**松弛固定法**

首先将变量$y$分割成$R$个bu的子集$Q^1,Q^2,\cdots,Q^R$，同时定义子集$U^r$，$U^r=\cup_{u=r+1}^RQ^u$。我们接着会求解$R$个序列MIP问题（$\text{MIP}^r, 1\leq r\leq R$），来寻找原始MIP问题的启发解。具体来说，$Q^1$表示变量$y$的下标为$\{1,2,\cdots,t_1\}$，$Q^2$表示变量$y$的下标为$t_1+1,\cdots,t_2$，$U^1$表示变量$y$的下标为$t_1+1,\cdots,u_1$。

序列问题$\text{MIP}^1$中，我们只考虑索引$Q^1\cup U^1$的整数变量，$Q$中其他索引的变量进行线性松弛，
$$
\begin{aligned}
(\text{MIP}^1)\quad\text{min}\quad \{cx+fy:\quad &Ax+By\geq b\\
&x\in\mathbb{R}_+^n\\
&y_j \in \{0,1\}& \text{for all}&\quad j\in Q^1\cup U^1\\
&y_j\in[0,1] & \text{for all} &\quad j \in Q\backslash(Q^1\cup U^1)
\}
\end{aligned}
$$
令$(x^1,y^1)$表示$\text{MIP}^1$的解，固定索引$Q^1$的值为$y^1$，接着进行$\text{MIP}^2$，对于之后的序列MIP问题，需要固定之前所有的最优解，表示如下
$$
\begin{aligned}
(\text{MIP}^r)\quad\text{min}\quad \{cx+fy:\quad &Ax+By\geq b\\
&x\in\mathbb{R}_+^n\\
&y_j=y_j^{r-1}\in\{0,1\}\quad\text{for all }j\in Q^1\cup\cdots\cup Q^{r-1}\\
&y_j \in \{0,1\},\quad \text{for all }j\in Q^r\cup U^r\\
&y_j\in[0,1],\quad \text{for all }j \in Q\backslash(Q^1\cup\cdots\cup Q^{r}\cup U^r)
\}
\end{aligned}
$$
$(x^r,y^r)$表示序列问题$\text{MIP}^r$的解。

如果所有的序列MIP问题都有解，那么最终可以得到一个可行的启发解，否则求解会终止在任意一步。

我们使用一个20个二元变量的案例描述具体的松弛固定法的过程，其中维度20可以理解为规划问题中的时间周期。在该问题中$R=4$，$Q^1=\{1,\cdots,5\}$，$Q^2=U^1=\{6,\cdots,10\}$，$Q^3=U^2=\{11,\cdots,15\}$，$Q^4=U^3=\{16,\cdots,20\}$。松弛固定法过程如下：

- 在$\text{MIP}^1$中，我们令下标$\{1,\cdots,10 \}$为整数变量（$Q^1\cup U^1$），松弛其他变量。
- 根据$\text{MIP}^1$的解，我们在$\text{MIP}^2$中固定下标为$\{1,\cdots,5\}$的变量（$Q^1$），然后令下标$\{6,\cdots,15\}$为整数变量（$Q^2\cup U^2$），松弛下标$\{16,\cdots,20\}$变量。
- 在$\text{MIP}^3$中，固定下标$\{1,\cdots,10\}$，令下标$\{11,\cdots,20\}$为整数变量，无松弛变量。
- 因为$Q^4=U^3$在这里，如果$\text{MIP}^3$有解，我们无需再进行$\text{MIP}^4$的求解。

求解过程，亦可以用表格表示如下

| Iteration | MIP            | Fixed Var        | Binary Var        | Relaxed Var       | Solution    |
| --------- | -------------- | ---------------- | ----------------- | ----------------- | ----------- |
| $r=1$     | $\text{MIP}^1$ | -                | $1\leq t\leq10$   | $11\leq t\leq 20$ | $(x^1,y^1)$ |
| $r=2$     | $\text{MIP}^2$ | $1\leq t\leq 5$  | $6\leq t\leq 15$  | $16\leq t\leq 20$ | $(x^2,y^2)$ |
| $r=3$     | $\text{MIP}^3$ | $1\leq t\leq 10$ | $11\leq t\leq 20$ | -                 | $(x^3,y^3)$ |

从该案例中，可以看出松弛固定法的基本原理，我们使用$Q^r$来逐步固定变量，同时使用$U^r$来在不同周期间形成过渡，避免部分求解的短时问题。

松弛固定法，又被称为时间分解法，这是因为求解过程的分解，与规划问题中按照时间的分解是完全一致的。我们在求解一个20个时间周期的问题时，先对前5个周期进行求解，再在此基础上求解后5个周期，这样逐步递进，直到求解完所有周期。

但是问题在于，我们会在求解过程中遇到不可行解，通常是因为后续周期的产能约束难以满足。一种可能的方式是为之前周期的库存水平添加下限约束，这样更容易使后续周期得到可行解。在没有固定生产消耗的情况下，这种下限库存水平很容易计算，否则的话只能对下限库存水平进行估计。

最后，类似的松弛固定法，也可以用于按照机器、产品类型或者地理区域的问题分解。

### 3.6.2 改进启发解

我们在有一个线性最优解$(\hat{x},\hat{y})$和整数最优解$(\bar{x},\bar{y})$的基础上，可以尝试在有限的时间内寻找其相邻解，以期获得更好的解，也就是改进启发解的基本思路。

**松弛诱导邻近搜索（RINS）**

RINS的思路使在线性最优解$(\hat{x},\hat{y})$和整数最优解$(\bar{x},\bar{y})$的邻域内搜索，如果变量$y_j$在线性解和整数解相同，那么固定该值，求解剩余的$\text{MIP}^{\text{RINS}}$
$$
\begin{aligned}
(\text{MIP}^{\text{RINS}})\quad\text{min}\quad\{cx+fy:\quad &Ax+By\geq b\\
&x\in \mathbb{R}_+^n\\
&y\in \{0,1\}^p\\
&y_j=\bar{y}_j \quad \text{for all}\quad j\in Q \text{ with } \hat{y}_j=\bar{y}_j
\}.
\end{aligned}
$$
RINS可以看作是切割固定法的改进版本。

**局部分支（LB）**

局部分支直接构造与整数最优解相似的整数解，定义整数$k$，构造解与整数最优解的最大偏差不超过$k$作为约束，形式如下
$$
\displaystyle
\begin{aligned}
(\text{MIP}^{\text{LB}})\quad\text{min}\quad\{cx+fy:\quad &Ax+By\geq b\\
&x\in \mathbb{R}_+^n\\
&y\in \{0,1\}^p\\
&\sum_{j\in Q:\bar{y}_j=0}y_j+\sum_{j\in Q: \bar{y}_j=1}(1-y_j)\leq k
\}.
\end{aligned}
$$
该方法显然有许多变种，例如只在一个子集$Q^*$内变化，或者只变化0的变量，或者与其他启发式方法结合等等。

**结论 3.7 **  *固定子集* $Q^*$ *内的约束* $y_j=\bar{y}_j, j\in Q^*$ *（如松弛固定或RINS）可以使用如下的局部分支约束表示*
$$
\displaystyle \sum_{j\in Q^*:\bar{y}_j=0}y_j+\sum_{j\in Q^*:\bar{y}_j=1}(1-y_j)\leq 0
$$

这也意味着，我们可以使用同样的形式构造在$k\geq  1$时松弛固定约束和RINS约束
$$
\displaystyle \sum_{j\in Q^*:\bar{y}_j=0}y_j+\sum_{j\in Q^*:\bar{y}_j=1}(1-y_j)\leq k
$$
**交换（EXCH）**

最后介绍松弛固定法的一种改进形式——交换（EXCH）。在松弛固定法中，有一系列$Q^r,U^r$，令$Q^r$（或者$Q^r\cup U^r$）为整数变量，使用整数最优解$(\bar{x},\bar{y})$来固定其他的整数变量，得到交换的MIP构造如下
$$
\begin{aligned}
(\text{MIP}^{\text{EXCH},r})\quad\text{min}\quad\{cx+fy:\quad &Ax+By\geq b\\
&x\in \mathbb{R}_+^n\\
&y_j=\bar{y}_j &\text{for all}\quad& j\in Q\backslash Q^r \\
&y_j \in \{0,1\} & \text{for all}\quad& j\in Q^r &\}.
\end{aligned}
$$
在该方法中，对于子集的计算顺序并无要求，每个序列MIP问题也都是相互独立的。

