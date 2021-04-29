# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:47:19 2020

@author: lefin
"""
from sympy import symbols, Eq, solve, Symbol, Rational, integrate, Mul, sqrt

x,y = symbols('x, y')

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

def negComb(n,r):
    if r == 0:
        return 1
    else:
        return (n-r+1)*negComb(n,r-1)

def comb(n, r):
    '''
    this function is a way to calculate combinations: n things taken r at a time without repetition.
    '''
    if n > 0:
        return (Rational(fac(n),fac(n-r)*fac(r)))
    else:
        return Rational(negComb(n,r),fac(r))


    

def pas(layer):
    '''
    this function returns the specific layer of the pascal triangle
    '''
    res = ''
    for x in range(layer+1):
        res += str(comb(layer, x)) + ' '
    return res

def listout(ls):
    lss = [f'{tidx}. {term}\n' for tidx,term in enumerate(ls,1)]
    return ''.join(lss)

class binomial():
    '''
    This is the function for a binomial expansion, where it expands a 
    binomial expression and returns the final set
    '''
    def __init__(self,x,y,n,firstTerms = 4, firstTermLimit = False):
        self.x = x
        self.y = y
        self.n = n
        if n >= 1:
            strExpr = []
            preExpr = []
            expr = []
            if firstTermLimit: #if normal binomial expressions need first term limits
                rnge = firstTerms
            else:
                rnge = n+1
            
            for k in range(rnge):
                strExpr.append(f'({n},{k}), {x}^({n-k}), {y}^({k})')
                ls = [comb(n,k), x**(n-k),y**(k)]
                for lidx,lls in enumerate(ls):
                    try: 
                        ls[lidx] = Rational(lls)
                    except:
                        ls[lidx] = lls
                preExpr.append(ls)
                expr.append(ls[0]*ls[1]*ls[2])
            self.strExpr = strExpr
            self.preExpr = preExpr
            self.expr = expr
        else:
            strExpr = []
            preExpr = []
            expr = []
            for k in range(firstTerms):
                strExpr.append(f'({n},{k}), {x}^({n-k}), {y}^({k})')
                ls = [comb(n,k), x**(n-k),y**(k)]
                for lidx,lls in enumerate(ls):
                    try: 
                        ls[lidx] = Rational(lls)
                    except:
                        ls[lidx] = lls
                preExpr.append(ls)
                expr.append(ls[0]*ls[1]*ls[2])
            self.strExpr = strExpr
            self.preExpr = preExpr
            self.expr = expr
    
    def __str__(self):
        return f'sum({self.n},k=0): C({self.n},k) * {self.x}^({self.n}-k) * {self.y}^(k) \n\n{listout(self.strExpr)}\n\n{listout(self.preExpr)}\n\n{listout(self.expr)}\n'

    '''def termPower(self,power):
        for term in self.expr:
            if power in term.as_ordered_factors():
                return term
    def coefPower(self,power):
        for term in self.expr:
            if power in term.as_ordered_factors():
                return term/power'''
    
    def termPower(self,power):
        return [term for term in self.expr if power in term.as_ordered_factors()]
                
            
    def coefPower(self,power):
        return [term/power for term in self.expr if power in term.as_ordered_factors()]

    
    def termPowerExt(self,power):
        return [(self.strExpr[tidx],self.preExpr[tidx],term) for tidx,term in enumerate(self.expr) if power in term.as_ordered_factors()]
            
    def coefPowerExt(self,power):
        return [(self.strExpr[tidx],self.preExpr[tidx],term/power) for tidx,term in enumerate(self.expr) if power in term.as_ordered_factors()]

#||NOTE|| the printed binomial structure is in descending
#         power of the first variable

#b = binomial(2*x**6,5/(x**3),10)
b = binomial(4,7*x,-5,10)
print(b)



    

