# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:52:17 2021

@author: lefin
"""

def append(lst,*args):
    lst += [n for n in args]

def Max(iterable):
    current_max = None
    for n in iterable:
        if current_max == None or n > current_max:
            current_max = n
    return current_max

def Min(iterable):
    current_min = None
    for n in iterable:
        if current_min == None or n < current_min:
            current_min = n
    return current_min

class Array():
    def __init__(self, *args):
        self.table = {}
        self.len = 0
        for idx,n in enumerate(args):
            self.table[idx] = n
            self.len += 1
        
    
    def __str__(self):
        return 'a'+[self.table[n] for n in self.table].__str__()
    
    def get(self, index):
        if index not in self.table:
            print('Invalid Index')
            return
        return self.table[index]
    
    def insert(self,item,index): #insert item at index
        for idx in range(index,self.len+1)[::-1]: #add one index at the end
            self.table[idx] = self.table[idx-1]
        self.table[index] = item
        self.len += 1
        
    def delete(self,index):
        for idx in range(index+1,self.len): #add one index at the end
            self.table[idx-1] = self.table[idx]
        del self.table[Max(self.table)]
        self.len -= 1
        
    def last(self):
        return self.table[Max(self.table)]
    
    def Max(self):
        return Max([self.table[n] for n in self.table])
    
    def append(self, item):
        self.table[self.len] = item
        self.len += 1
    

a = Array(1,2,3,4)
a.insert('a',1)
print(a)
a.delete(1)
print(a)
a.append(6)
print(a)