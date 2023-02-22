function UCS return {solution, failure}
    结点初始化
    优先级队列初始化，优先级按代价排列
    closed表初始化
    loop do 
        队列为空即失败
        取出队列首结点
        目标检测
        将该结点的状态存入closed表中
        for 状态转移 in 基于当前状态的状态转移表 do 
            创建子结点
            if 新状态未访问
                将新结点推入队列
            else if 新状态在访问队列中且拥有更高的代价
                进行结点替换
                
                
                
function DLS return {solution, failure/cutoff}
    return RECURSIVE-DLS(NewNode, problem, limit)

function RECURSIVE-DLS(ndoe, problem, limit) return {solution, failure/cutoff}
    目标检测
    else if limit=0 then return cutoff
    else
        初始化是否剪枝bool量为否
        for 状态转移 in 基于当前状态的状态转移表 do 
            创建子结点
            result = RECURSIVE-DLS(child, problem, limit-1)
            if result为剪枝结果 
                设置剪枝bool量为真
            else if result不为失败 返回结果
        if 已发生剪枝 
            return cutoff
        else 
            return failure
        

function IDS return {solution, failure}
    for depth=0 to inf do
        result = DLS(problem, result)
        if 不为剪枝结果 then return result
        
        
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b 
    return abs(x1-x2) + abs(y1-y2)
        
def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost 
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current 
    
    return came_from, cost_so_far

function RECURSIVE-BEST-FIRST-SEARCH return {solution, failure}
    return RBFS(problem, newNode(INITIAL_NODE), inf)

function RBFS(problem, node, f_limit) return {solution, failure, new f-cost limit}
    目标检测
    successors = {}
    for 状态转移 in 基于当前状态的状态转移表 do 
        创建结点放入successors中
    if successors为空 return {failure, inf}
    for 结点s in successors do 
        更新s.f，选择{s.g+s.h, node.f}其中更大的
    loop do 
        记录successors中当前代价最低的结点为best 
        if best.f > f_limit then return {failure, best.f}
        记录successors中当前代价第二低的结点为alternate
        result, best.f = RBFS(problem, best, min(f_limit, alternative))
        if 不失败 then return result