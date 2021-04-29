# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:05:25 2021

@author: lefin
"""
point1 = (4,1)
point2 = (2,3)

if point1[1]<point2[1]:
    a = list(range(point1[1]+1,point2[1]))
else:
    a = list(range(point2[1]+1,point1[1]))


if point1[0]<point2[0]:
    b = list(range(point1[0]+1,point2[0]))
else:
    b = list(range(point1[0]+1,point2[0]))


if b == []:
    b = [point1[0]]
c = list(zip(sorted((b * int(len(a)/len(b)))[:len(a)]),a))


print(a,b)
print(c)