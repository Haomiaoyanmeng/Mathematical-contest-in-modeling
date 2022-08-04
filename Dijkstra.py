#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Information:
 @author   : enrico
 @contact  : wooenrico@gmail.com
 @Site     :
 @software : PyCharm
 @file     : Dijkstra.py
 @time     : 2018/5/2 20:50
 @desc     :
 这个代码需要搭配https://www.bilibili.com/video/BV1zz4y1m7Nq?from=search&seid=13570034695580909942&spm_id_from=333.337.0.0
 来学习更好
"""
# 定义不可达距离
_ = float('inf')


# points点个数，edges边个数,graph路径连通图,start起点,end终点
def Dijkstra(points, edges, graph, start, end):
    # map = [[_ for i in range(points + 1)] for j in range(points + 1)]  # 没用就给他注释掉了
    pre = [0] * (points + 1)  # 记录前驱  就是这个点的前一个点hhh
    vis = [0] * (points + 1)  # 记录节点遍历状态  是否已经确定是最优路径了
    dis = [_ for i in range(points + 1)]  # 保存最短距离
    road = [0] * (points + 1)  # 保存最短路径
    roads = []
    map = graph  # 这里其实还是用了之前给的map

    for i in range(points + 1):  # 初始化起点到其他点的距离 这里其实就是第一次循环
        if i == start:
            dis[i] = 0
        else:
            dis[i] = map[start][i]
        if map[start][i] != _:
            pre[i] = start
        else:
            pre[i] = -1
    vis[start] = 1
    for i in range(points + 1):  # 每循环一次确定一条最短路, 循环完也就都确定了
        min = _  # 这里的min似乎没用哦哦哦, 有用有用, 这个是判断最小值的
        for j in range(points + 1):  # 寻找当前最短路
            if vis[j] == 0 and dis[j] < min:
                t = j
                min = dis[j]
        vis[t] = 1  # 找到最短的一条路径 ,标记
        for j in range(points + 1):  # 这里是根据新的t点进行扩展
            if vis[j] == 0 and dis[j] > dis[t] + map[t][j]:
                dis[j] = dis[t] + map[t][j]
                pre[j] = t
    p = end
    len = 0
    while p >= 1 and len < points:
        road[len] = p  # 顺序是从end一个一个到start
        p = pre[p]  # 这里的意思就是一个一个点的往前面找
        len += 1
    mark = 0
    len -= 1
    while len >= 0:  # 这里其实就是反过来存到roads里面
        roads.append(road[len])
        len -= 1
    return dis[end], roads


# 固定map图
def map():
    map = [[_, _, _, _, _, _],
           [_, _, 2, 3, _, 7],
           [_, 2, _, _, 2, _],
           [_, 3, _, _, _, 5],
           [_, _, 2, _, _, 3],
           [_, 7, _, 5, 3, _]
           ]
    s, e = input("输入起点和终点：").split()
    dis, road = Dijkstra(5, 7, map, int(s), int(e))
    print("最短距离：", dis)
    print("最短路径：", road)


# 输入边关系构造map图
def createmap():  # 这里其实是可以让一个一个输入那个边和点的值的, 但是实际不会这么去用
    a, b = input("输入节点数和边数：").split()
    n = int(a)
    m = int(b)
    map = [[_ for i in range(n + 1)] for j in range(n + 1)]
    for i in range(m + 1):
        x, y, z = input("输入两边和长度：").split()
        point = int(x)
        edge = int(y)
        map[point][edge] = float(z)
        map[edge][point] = float(z)
    s, e = input("输入起点和终点：").split()
    start = int(s)
    end = int(e)
    dis, road = Dijkstra(n, m, map, start, end)
    print("最短距离：", dis)
    print("最短路径：", road)


if __name__ == '__main__':
    map()
