# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:03:37 2020

@author: lefin
"""

import numpy as np
from timeit import timeit
from time import sleep
from copy import deepcopy
from wincon.WinCon import Screen, Float
import threading
from math import sqrt
'''
def nonz(ls):
    for n in ls:
        if n != 0:
            return 
'''

class MGraph(): #Matrix Graph / Adjacentcy Matrix
    def __init__(self, verticies: list, links: list, directed = False):
        self.vertDict = {}
        self.matrix = np.zeros((len(verticies),len(verticies)))
        self.directed = directed
        for vidx, vertex in enumerate(verticies):
            self.vertDict[vertex] = vidx
            self.vertDict[vidx] = vertex
        
        for link in links:
            ls = link.split(',')
            if len(ls) == 2:
                ls.append(1)
            self.matrix[self.vertDict[ls[0]]][self.vertDict[ls[1]]] = ls[2]
            if not directed:
                self.matrix[self.vertDict[ls[1]]][self.vertDict[ls[0]]] = ls[2]
    
    def full_bfs(self,start):
        self.bfs_ls = []
        def search(nodes):
            nodes = list(dict.fromkeys(nodes))
            self.bfs_ls += nodes
            ls = [self.vertDict[lidx] for node in nodes for lidx,link in enumerate(self.matrix[self.vertDict[node]]) if link and self.vertDict[lidx] not in self.bfs_ls]
            if ls != []:
                search(ls)
        search([start])
        return self.bfs_ls
    
    def full_dfs(self,start):
        self.dfs_ls = []
        dfsMatrix = deepcopy(self.matrix)
        #zero = np.zeros(dfsMatrix.shape[0])
        def search(node, parent_ls):
            #print(node,dfsMatrix[node],parent_ls)
            self.dfs_ls.append(self.vertDict[node])
            if np.all(dfsMatrix[node] == 0) :
                if parent_ls == []:
                    return
                search(parent_ls[-1],parent_ls[:-1])
            else:
                popped = np.where(dfsMatrix[node] > 0)[0][0]
                #print(popped)
                dfsMatrix[node][popped] = 0
                if not self.directed:
                    dfsMatrix[popped][node] = 0
                parent_ls += [node]
                search(popped,parent_ls)
        search(self.vertDict[start],[])
        return list(dict.fromkeys(self.dfs_ls))
    
    def __str__(self):
        return '\n'.join([f'{self.vertDict[ridx]} {row}'for ridx, row in enumerate(self.matrix)])
    
    def dijkstra(self,start):
        self.d_ls = []
        def search(node, dis, prev):
            self.d_ls.append((node,dis,prev))
            ls = [(self.vertDict[lidx],link+dis, node, dis) for node,dis,prev in self.d_ls for lidx,link in enumerate(self.matrix[self.vertDict[node]]) if link and self.vertDict[lidx] not in [u[0] for u in self.d_ls]]
            #print(ls)
            if len(self.d_ls) != len(self.vertDict)/2:
                search(*min(ls,key=lambda x:x[1])[:3])
        search(start,0,start)
        return MGraph([u[0] for u in self.d_ls],[f'{node},{prev},{self.matrix[self.vertDict[node]][self.vertDict[prev]]}' for node,dis,prev in self.d_ls],self.directed)
    
    def dijkstra_print(self,start):
        self.d_ls = []
        def search(node, dis, prev):
            self.d_ls.append((node,dis,prev))
            ls = [(self.vertDict[lidx],link+dis, node, dis) for node,dis,prev in self.d_ls for lidx,link in enumerate(self.matrix[self.vertDict[node]]) if link and self.vertDict[lidx] not in [u[0] for u in self.d_ls]]
            print(sorted(ls,key=lambda x:x[1]))
            if len(self.d_ls) != len(self.vertDict)/2:
                search(*min(ls,key=lambda x:x[1])[:3])
        search(start,0,start)
        return self.d_ls


class IMGraph(): #Integer Matrix Graph / Adjacentcy Matrix
    def __init__(self, nodes: int, links: list, directed = False):
        self.matrix = np.zeros((nodes,nodes))
        self.directed = directed
        for link in links:
            ls = list(map(int, link.split(',')))
            if len(ls) == 2:
                ls.append(1)
            self.matrix[ls[0]][ls[1]] = ls[2]
            if not directed:
                self.matrix[ls[1]][ls[0]] = ls[2]
    
    def full_bfs(self,start):
        self.bfs_ls = []
        def search(nodes):
            nodes = list(dict.fromkeys(nodes))
            self.bfs_ls += nodes
            ls = [lidx for node in nodes for lidx,link in enumerate(self.matrix[node]) if link and lidx not in self.bfs_ls]
            if ls != []:
                search(ls)
        search([start])
        return self.bfs_ls
    
    def full_dfs(self,start):
        self.dfs_ls = []
        dfsMatrix = deepcopy(self.matrix)
        #zero = np.zeros(dfsMatrix.shape[0])
        def search(node, parent_ls):
            #print(node,dfsMatrix[node],parent_ls)
            self.dfs_ls.append(node)
            if np.all(dfsMatrix[node] == 0) :
                if parent_ls == []:
                    return
                search(parent_ls[-1],parent_ls[:-1])
            else:
                popped = np.where(dfsMatrix[node] > 0)[0][0]
                #print(popped)
                dfsMatrix[node][popped] = 0
                if not self.directed:
                    dfsMatrix[popped][node] = 0
                parent_ls += [node]
                search(popped,parent_ls)
        search(start,[])
        return list(dict.fromkeys(self.dfs_ls))
    
    def __str__(self):
        return str(self.matrix)
    


class DGraph(): #Dictionary Graph / Adjacentcy List
    def __init__(self, verticies: list, links: list, directed = False, manualDirected = False, tuples = False, graphical = False):
        self.graphical = graphical
        self.vertDict = {}
        self.directed = directed
        self.tuples = tuples
        if not tuples:
            for vertex in verticies:
                if directed or manualDirected:
                    self.vertDict[vertex] = [[link.split(',')[1],eval(link.split(',')[2]),0] for link in links if vertex == link.split(',')[0]]
                else:
                    self.vertDict[vertex] = [[[x for x in link.split(',') if x != vertex][0], eval(link.split(',')[2]),0] for link in links if vertex in link.split(',')]
                self.vertDict[vertex].sort(key = lambda x:x[1])
        else:
            for vertex in verticies:
                if directed or manualDirected:
                    self.vertDict[vertex] = [[link[1],link[2],0] for link in links if vertex == link[0]]
                else:
                    self.vertDict[vertex] = [[[x for x in link if x != vertex][0], link[2],0] for link in links if vertex in link]
                self.vertDict[vertex].sort(key = lambda x:x[1])
    
    
    def getLink(self,node,dest):
        #print(node,dest)
        for link in self.vertDict[node]:
            if dest in link:
                #print(link)
                return link
    def getSubGrapgLink(self,node,dest,graph):
        for link in graph[node]:
            if dest in link:
                #print(link)
                return link
    
    def full_bfs(self, start):
        self.bfs_ls = []
        def search(nodes):
            nodes = list(dict.fromkeys(nodes)) #eliminate duplicates
            self.bfs_ls += nodes #append nodes
            #Generate the list of nodes to search for
            #Which are nodes that are linked to the searched node but not currently in the final list
            ls = [linked_node[0] for node in nodes for linked_node in self.vertDict[node] if linked_node[0] not in self.bfs_ls]
            if ls != []: #if there are still nodes that is avaliable, continue to search those nodes
                search(ls)
        search([start]) #initiate search
        return self.bfs_ls
    
    def full_dfs(self,start):
        self.dfs_ls = []
        dfsDict = deepcopy(self.vertDict) #create a copy of the adjacentcy list
        def search(node, parent_ls):
            #print(node,dfsDict[node],parent_ls)
            self.dfs_ls.append(node) #append the searched node
            if dfsDict[node] == []: #if there are no links within the seached node
                if parent_ls == []: #if there are no parent in the current path (this is the origin node)
                    return #end search
                search(parent_ls[-1],parent_ls[:-1]) #else, if there is parent but no links, return to parent node
            else: #if there are links within the searched node
                popped = dfsDict[node].pop(0) #pop the link between the searched node and linked node
                if not self.directed: #if the graph is not directed, delete also the returning link
                    popsub = popped[:]
                    popsub[0] = node
                    dfsDict[popped[0]].remove(popsub)
                parent_ls += [node] #parent list append current node as a parent of the current path
                search(popped[0],parent_ls)#go down the path and search the next node 
        search(start, [])
        return list(dict.fromkeys(self.dfs_ls))
    
    def __str__(self):
        return '\n'.join([f'{vert}: {self.vertDict[vert]}' for vert in self.vertDict])

    def removeNode(self,node):
        for links in self.vertDict[node]:
            self.vertDict[links[0]].remove([node,links[1],links[2]])
        self.vertDict.pop(node,None)
    
    def link(self,node,dest,length):
        self.vertDict[node].append([dest,length,0])
        if not self.directed:
            self.vertDict[dest].append([node,length,0])
    
    
    def dijkstra(self,start):
        self.d_ls = []
        self.visited = []
        def search(node, dis, prev):
            self.d_ls.append((node,dis,prev))#append the searched nodes and its data
            self.visited.append(node)
            #Generate The possible queue by going through the list of links of the searched node
            ls = [(link[0], link[1]+dis, node, dis) for node,dis,prev in self.d_ls for link in self.vertDict[node] if link[0] not in self.visited]
            
            if len(self.d_ls) != len(self.vertDict):
                search(*min(ls,key=lambda x:x[1])[:3])#search for the next node in the queue which is the one that has the minimum distance with the starting node
        search(start,0,start)
        
        return DGraph([u[0] for u in self.d_ls],[f'{node},{prev},{self.getLink(node,prev)[1]}' for node,dis,prev in self.d_ls[1:]],self.directed)

    def dijkstra_path(self,start,end): #dijkstra
        self.d_dict = {}
        self.visited = []
        def search(node, dis, prev):
            if end in self.visited:
                return
            self.d_dict[node] = (dis,prev)#establish seached node key in dictionary
            self.visited.append(node)#visited log
            ls = [(link[0], link[1]+self.d_dict[node][0], node, self.d_dict[node][0]) for node in self.d_dict for link in self.vertDict[node] if link[0] not in self.visited]
            #print(ls)
            if len(self.d_dict) != len(self.vertDict):
                search(*min(ls,key=lambda x:x[1])[:3])
        search(start,0,start)
        res = [end]
        while res[-1] != start:
            res.append(self.d_dict[res[-1]][1])#go back through the dijkstra dictionary to establish path
        
        return res[::-1]
            
        
    def Astar(self,start,end):
        if not self.tuples:
            return
        def distance(fromnode,tonode):
            return sqrt(sum([(t-f)**2 for f,t in zip(fromnode,tonode)]))
        self.astar_dict = {start:[None,0,distance(start,end)]} #node: (parent, local, global)
        self.astar_ls = [start]
        self.visited = []
        def search(node):
            #print(node)
            if end in self.astar_dict:
                return
            self.visited.append(node)
            local = self.astar_dict[node][1]
            for link in self.vertDict[node]:
                if (link[0] not in self.astar_ls or self.astar_dict[node][2] < self.astar_dict[link[0]][2]) and link[0] not in self.visited: #check if unvisited/node's local value smaller than link's/adjacent's local value
                    if link[0] not in self.astar_ls :
                        self.astar_ls.append(link[0])
                    self.astar_dict[link[0]] = [node,local+link[1],distance(link[0],end)]
            self.astar_ls.remove(node)#remove explored node
            self.astar_ls.sort(key=lambda x:self.astar_dict[x][1])#sort list by global value
            if self.astar_ls != []:
                search(self.astar_ls[0])
        search(start)
        #print(self.astar_dict[self.astar_dict[(2,0)][0]])
        def path(node):
            if self.astar_dict[node][0] == None:
                return [node]
            else:
                return [node] + path(self.astar_dict[node][0])
        
        return path(end)[::-1]

class DFullLattice(DGraph):
    def __init__(self,xd,yd,diagonals=False):
        self.xd = xd
        self.yd = yd
        nodes = [(r,c) for c in range(xd) for r in range(yd)]
        limitx, limity = xd-1,yd-1
        links = []
        
        if not diagonals:
            for node in nodes:
                if node[0] != 0:
                    links.append([node,(node[0]-1,node[1]),1])
                if node[0] != limity:
                    links.append([node,(node[0]+1,node[1]),1])
                if node[1] != 0:
                    links.append([node,(node[0],node[1]-1),1])
                if node[1] != limitx:
                    links.append([node,(node[0],node[1]+1),1])
        else: #Hey, I know it's stupid, but wtf, it works
            sqrt2 = sqrt(2)
            for node in nodes:
                T = [0,0,0,0]
                if node[0] != 0:
                    links.append([node,(node[0]-1,node[1]),1])
                    T[0] = 1
                if node[0] != limity:
                    links.append([node,(node[0]+1,node[1]),1])
                    T[1] = 1
                if node[1] != 0:
                    links.append([node,(node[0],node[1]-1),1])
                    T[2] = 1
                if node[1] != limitx:
                    links.append([node,(node[0],node[1]+1),1])
                    T[3] = 1
                
                if T[1] + T[3] == 2:
                    links.append([node,(node[0]+1,node[1]+1),sqrt2])
                if T[0] + T[3] == 2:
                    links.append([node,(node[0]-1,node[1]+1),sqrt2])
                if T[1] + T[2] == 2:
                    links.append([node,(node[0]+1,node[1]-1),sqrt2])
                if T[0] + T[2] == 2:
                    links.append([node,(node[0]-1,node[1]-1),sqrt2])
                
                
        super().__init__(nodes, links, tuples=True, manualDirected=True)
    
    
class VisualFullLattice(DFullLattice):
    def __init__(self,xd,yd,screen = None,diagonals=False, scaleFactor = 4):
        self.visited = []
        self.screen = screen
        self.sublists = []
        self.scaleFactor = scaleFactor
        super().__init__(xd,yd,diagonals)
        
    def link(self,node,dest,length):
        self.vertDict[node].append([dest,length,0])
        if not self.directed:
            self.vertDict[dest].append([node,length,0])
    
    def subListLink(self,node,dest,length,sublist,visited = 0):
        sublist[node].append([dest,length,visited])
        if not self.directed:
            sublist[dest].append([node,length,visited])
    
    def update(self, sublist = None): #main visual function
        
        if sublist == None:
            sublist = self.vertDict
        empty = [['઻' for x in range(self.xd*self.scaleFactor*2+self.scaleFactor*4+sum(range(self.xd+1))*2) ] for y in range(self.yd*self.scaleFactor+self.scaleFactor*2+sum(range(self.yd+1))) ]
        
        for nodeO in sublist:
            nodeS = 'o'
            if nodeO in self.visited:
                nodeS = 'x'
            node = (nodeO[1]*self.scaleFactor*2+self.scaleFactor+nodeO[1],nodeO[0]*self.scaleFactor+int(self.scaleFactor/2)+nodeO[0])
            for link in sublist[nodeO]:
                linkS = '઻'
                if link[2]:
                    linkS = '.'
                dest = (link[0][1]*self.scaleFactor*2+self.scaleFactor+link[0][1],link[0][0]*self.scaleFactor+int(self.scaleFactor/2)+link[0][0])
                point1 = node
                point2 = dest
                areverse = False
                breverse = False
                if point1[1]<point2[1]:
                    a = list(range(point1[1]+1,point2[1]))
                else:
                    a = list(range(point2[1]+1,point1[1]))[::-1]
                    areverse = True
                if a == []:
                    a = [point1[1]]
                
                if point1[0]<point2[0]:
                    b = list(range(point1[0]+1,point2[0]))
                else:
                    b = list(range(point2[0]+1,point1[0]))[::-1]
                    breverse = True
                if b == []:
                    b = [point1[0]]
                    
                if len(a)>len(b):
                    c = list(zip(sorted((b * int(len(a)/len(b)))[:len(a)],reverse=breverse),a))
                    
                else:
                    c = list(zip(b,sorted((a * int(len(b)/len(a)))[:len(b)],reverse=areverse)))
                
                '''
                print(node,dest,nodeO,(int(dest[1]/4),int(dest[0]/8)))
                print(a,b)
                print(c)
                print('') 
                '''
                
                for x,y in c:
                    empty[y][x] = linkS
            
            empty[node[1]][node[0]] = nodeS

        self.float = Float(empty, 0, 0, True)
        if self.screen != None:
            self.screen.setFloats([self.float])
        
    def full_bfs(self, start):
        sleep(0.5)
        self.visited = []
        self.bfs_ls = []
        def passThrough(node,link):
            link[2] = 1
            for link_link in self.vertDict[link[0]]:
                if node in link_link:
                    link_link[2] = 1
            return link[0]
        def search(nodes):
            sleep(0.5)
            nodes = list(dict.fromkeys(nodes))
            self.visited += nodes
            self.bfs_ls += nodes
            ls = [passThrough(node,linked_node) for node in nodes for linked_node in self.vertDict[node] if linked_node[0] not in self.bfs_ls]
            self.update()
            if ls != []:
                search(ls)
        search([start])
        return self.bfs_ls
    
    def full_dfs(self,start):
        sleep(0.5)
        self.visited = []
        self.dfs_ls = []
        dfsDict = deepcopy(self.vertDict)
        def passThrough(node,link):
            def linkIndict(link):
                for vert in self.vertDict[node]:
                    if link in vert:
                        return vert
            link = linkIndict(link)
            link[2] = 1
            for link_link in self.vertDict[link[0]]:
                if node in link_link:
                    link_link[2] = 1
            return link[0]
        
        def search(node, parent_ls):
            sleep(0.01)
            #print(node,dfsDict[node],parent_ls)
            self.dfs_ls.append(node)
            self.visited.append(node)
            if dfsDict[node] == []:
                if parent_ls == []:
                    return
                self.update()
                search(parent_ls[-1],parent_ls[:-1])
            else:
                popped = dfsDict[node].pop(0)
                if not self.directed:
                    popsub = popped[:]
                    popsub[0] = node
                    dfsDict[popped[0]].remove(popsub)
                    passThrough(node, popped[0])
                parent_ls += [node]
                self.update()
                search(popped[0],parent_ls)
        search(start, [])
        return list(dict.fromkeys(self.dfs_ls))
    
    def newSubList(self):
        self.sublists.append({})
        for node in self.vertDict:
            self.sublists[-1][node] = []
    
    
    def dijkstra(self,start):
        sleep(0.5)
        self.d_ls = []
        def passThrough(a,b,c,d,e):
            e[2] = 1
            for link_link in self.vertDict[a]:
                if c in link_link:
                    link_link[2] = 1
            return (a,b+d,c,e[1],d)
        def search(node, dis, prev, prevdis):
            #sleep(0.01)
            self.visited.append(node)
            self.d_ls.append((node,dis,prev,prevdis))#append the searched nodes and its data
            
            #Generate The possible queue by going through the list of links of the node
            queue = [passThrough(link[0], link[1], node, dis, link) for node,dis,prev,prevdis in self.d_ls for link in self.vertDict[node] if link[0] not in self.visited]
            #print(ls)
            self.update()
            if len(self.d_ls) != len(self.vertDict):
                search(*min(queue,key=lambda x:x[1])[:4]) #search for the next node in the queue which is the one that has the minimum distance with the starting node
        search(start,0,start,0)
        
        sleep(0.5)
        self.newSubList()
        for node,dist,prev,prevdis in self.d_ls[1:]:
            self.subListLink(node,prev,prevdis,self.sublists[-1])
        self.update(self.sublists[-1])
        #return DGraph([u[0] for u in self.d_ls],[f'{node},{prev},{self.getLink(node,prev)[1]}' for node,dis,prev in self.d_ls[1:]],self.directed)
    
    def dijkstra_path(self,start,end):
        sleep(0.5)
        self.d_dict = {}
        def passThrough(a,b,c,d,e):
            e[2] = 1
            for link_link in self.vertDict[a]:
                if c in link_link:
                    link_link[2] = 1
            return (a,b+d,c,e[1],d)
        def search(node, dis, prev, prevdis):
            if end in self.visited:
                return
            #sleep(0.01)
            self.visited.append(node)
            self.d_dict[node] = (dis,prev,prevdis)#append the searched nodes and its data
            
            #Generate The possible queue by going through the list of links of the node
            queue = [passThrough(link[0], link[1], node, self.d_dict[node][0], link) for node in self.d_dict for link in self.vertDict[node] if link[0] not in self.visited]
            #print(ls)
            self.update()
            if len(self.d_dict) != len(self.vertDict):
                search(*min(queue,key=lambda x:x[1])[:4]) #search for the next node in the queue which is the one that has the minimum distance with the starting node
        search(start,0,start,0)
        
        #self.d_dict.pop(start)
        
        sleep(0.5)
        self.newSubList()
        for node in self.d_dict:
            dist,prev,prevdis = self.d_dict[node]
            self.subListLink(node,prev,prevdis,self.sublists[-1])
        self.update(self.sublists[-1])
        
        sleep(1)
        #print(self.d_dict)
        res = [end]
        self.newSubList()
        while res[-1] != start:
            
            self.sublists[-1][res[-1]].append(self.getLink(res[-1], self.d_dict[res[-1]][1]))
            res.append(self.d_dict[res[-1]][1])#go back through the dijkstra dictionary to establish path
        self.update(self.sublists[-1])
        
    def Astar(self,start,end):
        sleep(0.5)
        if not self.tuples:
            return
        def distance(fromnode,tonode):
            return sqrt(sum([(t-f)**2 for f,t in zip(fromnode,tonode)]))
        self.astar_dict = {start:[None,0,distance(start,end)]} #node: (parent, local, global)
        self.astar_ls = [start]
        self.visited = []
        def search(node):
            
            self.update()
            #print(node)
            if end in self.astar_dict:
                return
            self.visited.append(node)
            local = self.astar_dict[node][1]
            for link in self.vertDict[node]:
                if (link[0] not in self.astar_ls or self.astar_dict[node][2] < self.astar_dict[link[0]][2]) and link[0] not in self.visited: #check if unvisited/node's local value smaller than link's/adjacent's local value
                    if link[0] not in self.astar_ls :
                        link[2] = 1
                        self.astar_ls.append(link[0])
                    self.astar_dict[link[0]] = [node,local+link[1],distance(link[0],end)]
            self.astar_ls.remove(node)#remove explored node
            self.astar_ls.sort(key=lambda x:self.astar_dict[x][1])#sort list by global value
            if self.astar_ls != []:
                search(self.astar_ls[0])
        search(start)
        #print(self.astar_dict[self.astar_dict[(2,0)][0]])
        def path(node):
            if self.astar_dict[node][0] == None:
                return [node]
            else:
                return [node] + path(self.astar_dict[node][0])
        
        sleep(1)
        #print(self.d_dict)
        res = path(end)
        self.newSubList()
        for node,nextnode in zip(res,res[1:]):
            self.subListLink(node,nextnode,self.getLink(node,nextnode)[1],self.sublists[-1],1) 
            
        self.update(self.sublists[-1])




s = Screen(fps=24, fullscreen=False)

fullL = VisualFullLattice(15,15,s,diagonals=True,scaleFactor=2)
fullL.removeNode((2,2))
fullL.removeNode((3,5))
fullL.update()
#print(fullL.Astar((0,1), (5,9)))
#print(fullL)

t = threading.Thread(target=fullL.Astar,args=[(0,1), (13,9)])
t.daemon = True



t.start()
s.start()
''''''



'''

fullL = DFullLattice(10,10,diagonals=True)


def a():
    fullL.Astar((0,1), (5,9))

def b():
    fullL.dijkstra_path((0,1), (5,9))

print(timeit(a,number = 100))
print(timeit(b,number = 100))
'''
'''
for r in range(9):
    for c in range(9):
        print(f'{r}{c} ',end='')
    print('')
'''

'''
M = MGraph(['A','B','C','D','AD'],['A,B,2','B,C,1','B,D,2','A,AD,20'])
D = DGraph(['u','v','x','w','y','z'],['u,v,1','u,w,2','v,w,2','v,x,2','w,x,2','y,z,2','y,w,3','x,y,3','w,z,3','x,z,1'])
M = MGraph(['u','v','x','w','y','z'],['u,v,1','u,w,2','v,w,2','v,x,2','w,x,2','y,z,2','y,w,3','x,y,3','w,z,3','x,z,1'])
IM = IMGraph(6,['0,1,1','0,3,2','1,2,2','1,2,2','3,2,2','4,5,2','4,3,3','2,4,3','3,5,3','2,5,1','1,3,2'])


print(D.dijkstra_path('u','y'))
#print(M.full_dfs('u'))
#print(fullL)


def g():
    M.full_bfs('u')
def dg():
    IM.full_bfs(0)

print(timeit(g))
print(timeit(dg))
'''