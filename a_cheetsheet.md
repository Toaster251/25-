sky 杨杰丞

格式比较混乱


练一练例题
滑动窗口，dsu，滚动数组，x对数幂x博弈，双dp，辅助栈
small
筛法，~~math中最小公倍数~~，~~更新最大最小~~？排队），取模理论，~~堆~~，前缀和，双队列双对双dp双指针双

### 2.HEAPQ
**最小堆**(heapq)可以维护列表中的最小值并将其位置放在第一个，即heap[0]。如果想得到最大值，以负值形式存入。
且最小堆通常涉及到内部元素的删除，而内置函数无此操作，则会利用到**懒删除**操作，使用字典记录已被删除的元素，需要取最小值时再一次性删除。
## 1.判断质数


代码

```python
def is_prime(n):  
    if n <= 1:  
        return False  
    for i in range(2, int(n**0.5) + 1):  
        if n % i == 0:  
            return False  
    return True
```

埃氏筛

```python
def sieve_of_eratosthenes(max_num):  
    """使用埃拉托斯特尼筛法找出 max_num 以内的所有素数"""  
    sieve = [True] * (max_num + 1)  
    sieve[0] = sieve[1] = False  # 0 和 1 不是素数  
    p = 2  
    while p * p <= max_num:  
        if sieve[p]:  
            for multiple in range(p * p, max_num + 1, p):  
                sieve[multiple] = False  
        p += 1  
    return [p for p, is_prime in enumerate(sieve) if is_prime]
```

欧式筛

代码

```python
def euler_sieve(n):
    is_prime = [True] * (n + 1)
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
    return primes
```

## 2.二分查找

往往写一个check函数后再在主函数中mid改变。（自己写）
while《
左加1右不变
（自己练一遍）

代码

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid  # 返回目标元素的索引
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```
对bisect库的使用
代码

```python
def bisect_left(arr, target):
    left, right = 0, len(arr)
    
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left
```

## 3.搜索

```python

```
```python
from collections import deque
  
def bfs(start, end):    
    q = deque([(0, start)])  # (step, start)
    in_queue = {start}


    while q:
        step, front = q.popleft() # 取出队首元素
        if front == end:
            return step # 返回需要的结果，如：步长、路径等信息

        # 将 front 的下一层结点中未曾入队的结点全部入队q，并加入集合in_queue设置为已入队
  
```

下面是对该模板中每一个步骤的说明,请结合代码一起看:

① 定义队列 q，并将起点(0, start)入队，0表示步长目前是0。 ② 写一个 while 循环，循环条件是队列q非空。 ③ 在 while 循环中，先取出队首元素 front。 ④ 将front 的下一层结点中所有**未曾入队**的结点入队，并标记它们的层号为 step 的层号加1，并加入集合in_queue设置为已入队。 ⑤ 返回 ② 继续循环。

为了防止走回头路，一般可以设置一个set类型集合in_queue来记录每个位置是否在BFS中已入过队。再强调一点，在BFS 中设置的 in_queue 集合的含义是判断结点是否已入过队，而不是**结点是否已被访问**。区别在于：如果设置成是否已被访问，有可能在某个结点正在队列中（但还未访问）时由于其他结点可以到达它而将这个结点再次入队，导致很多结点反复入队，计算量大大增加。因此BFS 中让每个结点只入队一次，故需要设置 in_queue 集合的含义为**结点是否已入过队**而非结点是否已被访问。




深搜与递归与回溯
## 1.dfs
dfs 如果要解决枚举类的题⽬通常会涉及回溯操作，⽽在原地修改时可能⽆需回溯。如果有回溯操 作必须要有退出条件。 防⽌递归深度过⼤，可以这样调整递归深度：
import sys sys.setrecursionlimit(1 << 30) 
如果 dfs 内部有类似于 dp 数组需要不断访问某些元素的值的时候，除了开空间创建⼀个 dp ，还可 以⽤ lru_cache 。 但⼀定要在需要进⾏记忆化递归的函数头顶上写，否则⽆效。 
‘’‘
from functools import lru_cache 
@lru_cache(maxsize=2048) # 或者更大，如None ，考虑内存因素自行调整 
def dfs()
’‘’

## 4.单调栈模板


```python

def monotonic_stack_template(arr):
    """单调栈通用模板"""
    n = len(arr)
    stack = []  # 存储索引，维护单调性
    result = [0] * n  # 根据问题调整初始化
    
    for i in range(n):
        # 破坏单调性时的处理（核心）
        while stack and arr[i] > arr[stack[-1]]:  # 根据问题选择 > 或 <
            # 找到符合条件的元素，进行计算
            idx = stack.pop()
            # 根据具体问题计算结果
            # result[idx] = ... 
        
        # 入栈前的处理（可选）
        if stack:
            # result[i] = stack[-1]  # 比如找左边第一个更大元素
        
        stack.append(i)  # 索引入栈
    
    # 处理栈中剩余元素（可选）
    # while stack: ...
    
    return result
```

维护递减
以及一个重要思想储存index而非数，
仅用数去条件判断
![[Pasted image 20251222120456.png]]
![[Pasted image 20251222120511.png]]

## 5.位运算
![[Pasted image 20251221004824.png]]
相应的用位运算作为储存的一些操作
![[Pasted image 20251221005200.png]]


## 6.菜鸟教程
| math.lcm()                                                                  | # 最小公倍数                                     |
| --------------------------------------------------------------------------- | ------------------------------------------- |
| [math.gcd()](https://www.runoob.com/python3/ref-math-gcd.html)              | 返回给定的整数参数的最大公约数。                            |
|                                                                             |                                             |
| [math.exp(x)](https://www.runoob.com/python3/ref-math-exp.html)             | 返回 e 的 x 次幂，Ex， 其中 e = 2.718281... 是自然对数的基数 |
|                                                                             |                                             |
|                                                                             |                                             |
| [math.factorial(x)](https://www.runoob.com/python3/ref-math-factorial.html) | 返回 x 的阶乘。 如果 x 不是整数或为负数时则将引发 ValueError。    |
|                                                                             |                                             |
|                                                                             |                                             |
| [math.ceil(x)](https://www.runoob.com/python3/ref-math-ceil.html)           | 将 x 向上舍入到最接近的整数                             |
| abs()                                                                       | 绝对值                                         |

（1）进制转换
1使用自带函数
十转二bin(),八oct()十六hex()
二转十int(num,2)

（2）asc2

在 Python 中，可以快速查看 ASCII 表：

```python
import string
print(string.printable)
print(repr(string.printable))
print(string.ascii_lowercase)
print(string.ascii_uppercase)
```
48–57 表示数字 0–9，65–90 表示大写字母 A–Z，97–122 表示小写字母 a–z。只要记住**大写字母的编码在小写字母之前**，就能大致推断出字符对应的十进制值。
ord(),chr()
## 7.dp
状态压缩？就是位运算。n一般小于20（1e7数量级）

dp的特征
最优 子结构，重叠 子问题，
kadane算法

```python
def kadane(arr):
    max_current = max_global = arr[0]
    for i in range(1, len(arr)):
        max_current = max(arr[i], max_current + arr[i])
        max_global = max(max_global, max_current)
    return max_global
#二维
def max_submatrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')

    for left in range(cols):
        temp = [0] * rows
        for right in range(left, cols):
            for row in range(rows):
                temp[row] += matrix[row][right]
            max_sum = max(max_sum, kadane(temp))
    return max_sum
```

经典二维dp

背包



## 8前缀和
一维

```python
    pre = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        pre[i + 1] = pre[i] + arr[i]
        
    result= prefix[r + 1] - prefix[l]
```


二维
```python
def build_prefix_2d(mat):
    """构建二维前缀和矩阵"""
    m, n = len(mat), len(mat[0])
    pre = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            pre[i + 1][j + 1] = mat[i][j] + pre[i][j + 1] + pre[i + 1][j] - pre[i][j]
    return pre

def query_2d(pre, x1, y1, x2, y2):
    """查询子矩阵[(x1,y1)到(x2,y2)]的和"""
    return pre[x2 + 1][y2 + 1] - pre[x1][y2 + 1] - pre[x2 + 1][y1] + pre[x1][y1]
```
## 9.dsu/并查集
与前缀树

这是路径压缩
```python
def find(i):
    if Parent[i] == i:
        return i
    else:
        result = find(Parent[i])  # 递归查找根
        Parent[i] = result        # 路径压缩：将节点直接连接到根
        return result
```
合并操作
一般用秩，
两者目的不同。
```python

# 合并（带按秩合并）
def union(a, b):
    root_a = find(a)
    root_b = find(b)
    
    if root_a == root_b:
        return  # 已经在同一集合中
    
    # 按秩合并：将秩小的树合并到秩大的树上
    if Rank[root_a] < Rank[root_b]:
        Parent[root_a] = root_b
    elif Rank[root_a] > Rank[root_b]:
        Parent[root_b] = root_a
    else:
        # 秩相等时，任意合并，但秩要加1
        Parent[root_b] = root_a
        Rank[root_a] += 1
        
def union_by_size(a, b, parent, size):
    root_a = find(a, parent)
    root_b = find(b, parent)
    
    if root_a == root_b:
        return
    
    # 按大小合并：将小集合合并到大集合
    if size[root_a] < size[root_b]:
        parent[root_a] = root_b
        size[root_b] += size[root_a]  # 更新大小
    else:
        parent[root_b] = root_a
        size[root_a] += size[root_b]  # 更新大小
```

题目：食物链 并查集, [http://cs101.openjudge.cn/practice/01182](http://cs101.openjudge.cn/practice/01182)

看不懂喵
一些奇妙的例题。

## 10区间
![[Pasted image 20251225113151.png]]

## 11滑动窗口
[滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)
给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。
```python

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        q = collections.deque()
        for i in range(k):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)

        ans = [nums[q[0]]]
        for i in range(k, n):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)
            while q[0] <= i - k:
                q.popleft()
            ans.append(nums[q[0]])
        
        return ans

作者：力扣官方题解
链接：https://leetcode.cn/problems/sliding-window-maximum/solutions/543426/hua-dong-chuang-kou-zui-da-zhi-by-leetco-ki6m/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

蘑菇
```python
def main():
    n, k = map(int, input().split())
    colors = list(map(int, input().split()))
    
    # 滑动窗口统计每个颜色出现的次数
    count = {}
    left = 0
    result = 0
    
    for right in range(n):
        # 将右边界颜色加入窗口
        count[colors[right]] = count.get(colors[right], 0) + 1
        
        # 如果颜色种类超过k，移动左边界
        while len(count) > k:
            count[colors[left]] -= 1
            if count[colors[left]] == 0:
                del count[colors[left]]
            left += 1
        
        # 以right为结尾的满足条件的子数组个数为 (right-left+1)
        result += right - left + 1
    
    print(result)

if __name__ == "__main__":
    main()
```


## 4.lambda函数
```python
sort() #--> 稳定的从小到大排序，如果列表存储的是多元元组，则依次按照每个元组的元素进行排序，且稳定
#如果想自行按照元组的元素顺序排序，可以使用lambda函数
s=[(1,2),(3,1),(4,5),(2,5)]
#按照第二个元素排序
s.sort(key=lambda x:x[1]) #[(3, 1), (1, 2), (2, 5)]
#按照第二个元素为首要升序排序，第一个元素为次要升序排序
s.sort(key=lambda x:(x[1],x[0])) #[(3, 1), (1, 2), (2, 5), (4, 5)]
#按照第二个元素为首要降序排序，第一个元素为次要升序排序
s.sort(key=lambda x:(-x[1],x[0])) #[(2, 5), (4, 5), (1, 2), (3, 1)]
#-----------------------------#
#如果想对数字按照字典序组合排序，得到最大最小整数，可以冒泡可以匿名
s=[9989,998]
#冒泡
for i in range(len(s)-1):
    for j in range(len(s)-i-1):
        if str(s[j])+str(s[j+1])<str(s[j+1])+str(s[j]):
            s[j],s[j+1]=s[j+1],s[j]
#lambda函数
s=sorted(s,key=lambda x: str(x)*10,reverse=True)
#-----------------------------#
#对字典的键值对进行排序，与列表存储元组差不多
d={3:34,2:23,9:33,10:33}
dd=dict(sorted(d.items(),key=lambda x:(x[1],-x[0]))) #{2: 23, 10: 33, 9: 33, 3: 34}
```
## V.SEARCHING  copy from汤伟杰

#### 1.dfs
dfs如果要解决枚举类的题目通常会涉及回溯操作，而在原地修改时可能无需回溯。如果有回溯操作必须要有退出条件。
防止递归深度过大，可以这样调整递归深度：
```python
import sys
sys.setrecursionlimit(1 << 30)
```
如果dfs内部有类似于dp数组需要不断访问某些元素的值的时候，除了开空间创建一个dp，还可以用lru_cache。
但一定要在需要进行记忆化递归的函数头顶上写，否则无效。
```python
from functools import lru_cache
@lru_cache(maxsize=2048) #或者更大，如None，考虑内存因素自行调整
def dfs():
    ...
```
1. 无回溯操作
例题：oj-lake counting-02386，原地修改
```python
dx=[-1,0,1,-1,1,-1,0,1]
dy=[-1,-1,-1,0,0,1,1,1]
def dfs(x,y):
    m[x][y]='.'
    for k in range(8):
        nx=x+dx[k]
        ny=y+dy[k]
        if 0<=nx<=n-1 and 0<=ny<=s-1 and m[x][y]=='W':
            dfs(nx,ny)
```
2. 有回溯操作
模板是：①有退出条件 ②递归之间做重复要做的事情 ③递归之后回溯为原状态
```python
def dfs():
    if ...:
        return 
    #do something
    dfs()
    #traceback
```
例题：
```python
#oj-八皇后-02754
'''考虑以下递归步骤：
在某次递归时，curr = [1, 5, 8, 6]，此时 ans.append(curr)。
接下来，回溯修改了 curr，变为 [1, 5, 8, 7]。
由于 ans 中保存的是 curr 的引用，ans 中原本存储的 [1, 5, 8, 6] 也会变为 [1, 5, 8, 7]。
因此使用 curr[:]，创建当前列表的拷贝，确保后续对 curr 的修改不会影响已保存的解
'''
visited=[0]*8
ans=[]
def dfs(k,curr):
    global ans
    if k==9:
        ans.append(curr[:])
        return
    for i in range(1,9):
        if visited[i-1]:
            continue
        if any(abs(j-i)==abs(len(curr)-curr.index(j)) for j in curr):
            continue
        visited[i-1]=1
        curr.append(i)
        dfs(k+1,curr)
        visited[i-1]=0
        curr.pop()
dfs(1,[])
# print(ans)
for _ in range(int(input())):
    n=int(input())
    print(''.join(map(str,ans[n-1])))
```
```python
#oj-有界的深度优先搜索-23558
def dfs(n,m,l,s,ans,k):
    if k==l+1 or s not in d:
        return
    for i in d[s]:
        if not visited[i] and i not in ans:
            visited[i]=1
            ans.append(i)
            dfs(n,m,l,i,ans,k+1)
            visited[i]=0

n,m,l=map(int,input().split())
d={}
for _ in range(m):
    a,b=map(int,input().split())
    if a>b: a,b=b,a
    if a not in d:
        d[a]=[]
    d[a].append(b)
    if b not in d:
        d[b]=[]
    d[b].append(a)

for v in d.values():
    v.sort()

s=int(input())
visited=[0]*n
ans=[s]
visited[s]=1
dfs(n,m,l,s,ans,1)
print(*ans)
```
---
#### 2.BFS
逐层扩展，用来求最小步数，模板；如果想保留路径，可以把路径作为参数传递，其中双端队列q加入的元素可能是三维，包含坐标和时间或者步数或者路径等等。
```python
from collections import deque
dx,dy=[0,-1,1,0],[-1,0,0,1]
def bfs(x,y,final):
    q=deque()
    q.append((x,y))
    inq=set()
    inq.add((x,y))
    step=1
    while q:
        for _ in range(len(q)): #遍历这一层，可以不写这一行，但是写了更清晰
            x,y=q.popleft()
            for i in range(4):
                nx,ny=x+dx[i],y+dy[i]
                if s[nx][ny]==final:
                    return step
                if 0<=nx<n and 0<=ny<m and s[nx][ny]==1 and (nx,ny) not in inq:
                    q.append((nx,ny))
                    inq.add((nx,ny))
        step+=1
    return None
```
例题：
```python
#oj-体育游戏跳房子-27237
#deque中多加入一个path不断传递
from collections import deque
def bfs(n,m,path):
    step=1
    q=deque()
    q.append((n,path))
    inq=set()
    inq.add(n)
    while q:
        for _ in range(len(q)):
            x,path=q.popleft()
            if x*3>0:
                if x*3==m:
                    return step,path+['H']
                if x*3 not in inq:
                    q.append((x*3,path+['H']))
                    inq.add(x*3)
            if x//2>0:
                if x//2==m:
                    return step,path+['O']
                if x//2 not in inq:
                    q.append((x//2,path+['O']))
                    inq.add(x//2)
        step+=1

while True:
    n,m=map(int,input().split())
    if {n,m}=={0}:
        break
    step,path=bfs(n,m,[])
    print(step)
    print(''.join(path))
```
---
#### 3.Dijkstra算法
解决单源最短路径问题，用于非负权图，使用`heapq`的最小堆来代替`bfs`中的`deque`，设置`dist`列表更新最短距离。
例题：
```python
#oj-走山路-20106
import heapq
dx,dy=[0,-1,1,0],[-1,0,0,1]
def dijkstra(sx,sy,ex,ey):
    if s[sx][sy]=='#' or s[ex][ey]=='#':
        return 'NO'
    q=[]
    dist=[[float('inf')]*m for _ in range(n)]
    heapq.heappush(q,(0,sx,sy)) #(distance,x,y)
    dist[sx][sy]=0
    while q:
        curr,x,y=heapq.heappop(q) #heappop()
        if (x,y)==(ex,ey):
                return curr

        for i in range(4):
            nx,ny=x+dx[i],y+dy[i]
            if 0<=nx<n and 0<=ny<m and s[nx][ny]!='#':
                new=curr+abs(s[x][y]-s[nx][ny])
                if new<dist[nx][ny]:
                    heapq.heappush(q,(new,nx,ny)) #heappush()
                    dist[nx][ny]=new
    return 'NO'
