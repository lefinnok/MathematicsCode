# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:47:19 2020

@author: lefin
"""
from sympy import symbols, Eq, solve, Symbol, Rational, integrate, Mul

x = symbols('x')

class nomin():
    def __init__(self, coef, power, char = 'x'):
        self.coef = coef
        self.power = power
        self.char = char

def fac(val):
    '''
    Factorial of an integer is the multiple between all the decending value 
    until 1
    '''
    val = max(int(val), 1)
    if val == 1:
        return 1
    else:
        return val*fac(val-1)

def comb(n, r):
    '''
    this function is a way to calculate combinations: n things taken r at a time without repetition.
    '''
    return (Rational(fac(n),fac(n-r)*fac(r)))

def pas(layer):
    '''
    this function returns the specific layer of the pascal triangle
    '''
    res = ''
    for x in range(layer+1):
        res += str(comb(layer, x)) + ' '
    return res

def binomial(x,y,n):
    '''
    This is the function for a binomial expansion, where it expands a 
    binomial expression and returns the final set
    '''
    if n >= 1:
        expr = []
        for k in range(n+1):
            ls = [comb(n,k), x**(n-k),y**(k)]
            for lidx,lls in enumerate(ls):
                try: 
                    ls[lidx] = Rational(lls)
                except:
                    ls[lidx] = lls
            expr.append(ls[0]*ls[1]*ls[2])
        return expr
    else:
        expr = []
        for k in range(4):
            ls = [comb(n,k), x**(n-k),y**(k)]
            for lidx,lls in enumerate(ls):
                try: 
                    ls[lidx] = Rational(lls)
                except:
                    ls[lidx] = lls
            expr.append(ls[0]*ls[1]*ls[2])
        return expr
    

def termPower(binomial,power):
    for term in binomial:
        if power in term.as_ordered_factors():
            return term
def coefPower(binomial,power):
    for term in binomial:
        if power in term.as_ordered_factors():
            return term/power

b = binomial(1,-2*x,-1)
print(coefPower(b,x**2))
    


    
    

