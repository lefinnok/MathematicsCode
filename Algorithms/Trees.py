# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:24:15 2021

@author: lefin
"""
from math import sqrt

class binary_tree:
    def __init__(self, lst):
        
        # it works, so, ehhhhh
        n = len(lst)
        if n <= 1:
            n = 0
        else:
            n += 1
        
        self.depth = int(sqrt(n)) + 1
        self.tree = []
        for level in range(self.depth):
            self.tree += [None] * 2**level
        
        #simplest implementation
        self.tree[:len(lst)] = lst
    
    
    def get_level(self,level):
        if level == 0:
            return [self.tree[0]]
        return self.tree[(2**level)-1:((2**level)-1)*2+1]
    
    def __str__(self):
        for level in range(b.depth):
            print(b.get_level(level))


b = binary_tree([5,2,7,1,6,2,4,5,1,1,1,1,1,1,1,2])
#b = binary_tree([0])

