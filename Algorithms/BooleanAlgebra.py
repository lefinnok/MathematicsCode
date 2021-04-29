# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:57:58 2021

@author: lefin
"""
import itertools


expression = "A'BC+AB'C+ABC'+ABC"

def eval_bool(expr:str):
    VARS = []
    for char in list(expr):
        if char not in ['+', "'"] + VARS:
            VARS.append(char)
    
    split_expression = expression.split('+') #split to terms
    split_terms = [list(term) for term in split_expression] #split terms to chars 
    compiled_terms = [[var + next_var if next_var == "'" else var for var,next_var in zip(term+ [''],term[1:]+ [''])] for term in split_terms] #combine not to previous char
    terms = [[var for var in term if var != "'"]for term in compiled_terms] #eliminater singled out nots (')
    for term in terms:
        print(term)
    

def binary_range(bits):
    bi = [[0,1]] * bits
    for combination in itertools.product(*bi):
        yield combination
    
def min_term_truth(bits,minterm_list):
    bi = [[0,1]] * bits
    for idx,combination in enumerate(itertools.product(*bi)):
        out = 0
        if idx in minterm_list:
            out = 1
        yield combination,out

def ftoMultiplex(i0,i1,i2,i3,s1,s0):
    if s1 == 0:
        if s0 == 0:
            return i0
        if s0 == 1:
            return i1
    if s1 == 1:
        if s0 == 0:
            return i2
        if s0 == 1:
            return i3

def ttoMultiplex(i0,i1,s):
    if s == 0:
        return i0
    else:
        return i1

    
print('j','k','Q',' ','D')
for a,b,c in binary_range(3):
    print(a,b,c,' ',int((a and not c) or (not b and c)),int(ttoMultiplex(a,not b,c)))

#for (a,b,c,d),out in min_term_truth(4, [0,1,3,4,11,13,14,15]):
    
    #print(a,b,c,d,' ',out,' ',int(ftoMultiplex(not c or d,not (c or d),c and d, c or d, a, b)))
'''
for idx,(a,b,c,d) in enumerate(binary_range(4)):
    print(idx,' ',a,b,c,d, (b and not c) or (a and not b) or (not b and c) ,not(not(b and not c)and not(a and not b) and not(c and not b)))
'''
'''
for a,b,c in binary_range(3):
    print(a,b,c,' ',ftoMultiplex(not a, 0, a, 1, b, c))
'''
'''
for idx,(a,b,c,d,e) in enumerate(binary_range(5)):
    print(idx,' ',a,b,c,d,e)
'''
'''
for a,b,c,d in binary_range(4):
    print(a,b,c,d,'  ->  ',int(not(a or b or c or d)), a, int(not(not(a or b or c or d) or a)))

'''