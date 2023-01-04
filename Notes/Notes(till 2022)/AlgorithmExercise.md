# 算法作业 2022-11-28

author：222070 仇隽

## HW2 算法导论 p404 15-1

题目：15-1 **Longest simple path in a directed acyclic graph**
Suppose that we are given a directed acyclic graph $G = (V,E)$ with real-valued edge weights and two distinguished vertices $s$ and $t$. Describe a dynamic-programming approach for finding a longest weighted simple path from $s$ to $t$.
What does the subproblem graph look like? What is the efficiency of your algorithm?

思路：DAG设计如图。
build array(3,7)，line1 as the node sequence, line2 as the latest time of arriving a node, line3 as the earlest time of arriving a node.

line2 computing method: initialize node_s(the beginning node) to be 0, visit the DAG to get the longest weight path towards the succeed, such as node0, iterate the whole DAG to finish this line.
line3 computing method: initialize node_t(the ending node in line2) with the result from line2, revisited the DAG to get the smallest weighted path towards the predecessor, such as node4, iterate the whole DAG to finish the line3.
Once finish the two line, compare the values. if equal, then push the node into the queue.
The path consisting of these nodes is the longest weighted path.

![解法](/Notes/pics/Q2.PNG)

## HW3 The Travel around Canada

该问题即算法竞赛书中的旅行问题
![旅行问题](/Notes/pics/Tour.png)

Assumption as follows:

     因为不能重复，所以每次假设仅往前移动一步；
     由于对称性，故令 j < i。


1. 建立关于所有结点之间的距离关系Matrix：dis[i][j]，value为两点间欧氏距离，其中ij满足$1 <= j <= i <= n$。
```c++
dis[i][j] = sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j])*(y[i] - y[j]));
```
2. 定义d[i][j]表示从起点n开始的两人一个到i点一个到j点的路径和的最小值；
边界情况为：
```c++
d[n-1][j] = dis[n-1][n] + dis[j][n];
```
表示最初刚出发时的情况。这里有两种写法，第一种是以序号1的node为起点，一种是以序号n的node为起点，两者的区别仅在于方向不同，导致for loop的i有自增自减，最终结果为$d[2][1]+ dis[1][2]$ 还是 $d[n-1][n] + dis[n-1][n]$，并没有本质区别。
这里我们采用以序号n为起点的算法。两人一个到n-1点（即走出一步），另一个走到j，j在(1,n-1)范围内更新d状态，也即两点间(j, n)的欧氏距离。

3. 状态转移方程
$min(dis[i][i + 1] + d[i + 1][j], dis[j][i + 1] + d[i + 1][i]);$
很容易理解，也就是判断新的点由谁来走，谁最短归谁，其中$dis[i][i + 1] + d[i + 1][j]$表示在移动到i+1和j点的情况下由i+1点走向i，$dis[j][i + 1] + d[i + 1][i]$表示在移动到i+1和i点的情况下由i+1点走向j。
参考代码如下：
```c++
for(int i = n - 1; i >= 2; --i)
	for (int j = 1; j < i; ++j) {
		if (i == n - 1) d[i][j] = dis[i][n] + dis[j][n];
		else d[i][j] = 
            min(dis[i][i + 1] + d[i + 1][j], dis[j][i + 1] + d[i + 1][i]);
	}
```
以下是示意图中$d[2][1]$的情况，黄色线路为$dis[2][3] + d[3][1]$，表示走的方向为在走到了3和1的基础上由3走向2；绿色线路为$dis[1][3] + d[3][2]$，表示在走到了3和2的基础上由3走向1。
![示意图](/Notes/pics/route.PNG)