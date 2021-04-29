# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:57:51 2021

@author: lefin
"""
from sympy import symbols, ln, E, simplify, log, sin, cos, tan, sec, exp, csc, cot, asin, acos, atan, solve
import sympy as sym

x,y = symbols('x,y')
log_type = type(log(x,2).args[1])
e = E

def differentiate(expression, times = 1):
    for _ in range(times-1):
        expression = differentiate(expression)
    ###rule of constant
    # if expression is a constant
    if expression.is_number: 
        return 0
    ###when expression is a symbol with a power of 1
    elif type(expression) == sym.Symbol:
        return 1
    
    ###sum/difference rule
    elif type(expression) == sym.Add:
        res = 0
        for arg in expression.args:
            #print(arg,differentiate(arg))
            res += differentiate(arg)
        #print(res)
        return res
    
    ###multiple / product rule / quotient rule (when constant, one of the term will be 0)
    #sympy likes its multiples, so it expresses every thing in it
    elif type(expression) == sym.Mul:
        #logarithms (with base)
        if type(expression.args[1]) == log:
            if hasattr(expression.args[0],'args'):
                if expression.args[0].args[0] == log:
                    base = expression.args[0].args[0].args[0]
                    inner_term = expression.args[1].args[0]
                    #chain rule
                    if type(inner_term) != sym.Symbol: 
                        return 1/(inner_term*ln(base)) * differentiate(inner_term)
                    return 1/(inner_term*ln(base)) 
        
        #quotient
        if type(expression.args[0]) == sym.Pow and expression.args[0].args[1] == -1: #this is the way sympy represents fractions, with a power of -1
            f = expression.args[1]
            g = expression.args[0].args[0]
            return (g*differentiate(f) - f*differentiate(g))/(g**2)
        #product
        else:
            f = expression.args[1]
            g = expression.args[0]
            return g*differentiate(f)+f*differentiate(g)
    
    ###power rule / exponent(non-natural)
    elif type(expression) == sym.Pow:
        power = expression.args[1]
        inner_term = expression.args[0]
        
        ##exponent
        if inner_term.is_number:
            inner_term = expression.args[1]
            base = expression.args[0]
            #chain rule
            if type(inner_term) != sym.Symbol:
                return expression*ln(base)*differentiate(inner_term)
            return expression*ln(base)
        
        ##power
        else:
            #chain rule
            if type(inner_term) != sym.Symbol:
                return (power*inner_term**(power-1))*differentiate(inner_term)
            return power*inner_term**(power-1)
    
    ###logarithm (without base)
    elif type(expression) == sym.log: #log = plain ln, or log base e
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol: 
            return (1/inner_term) * differentiate(inner_term)
        return 1/inner_term
    
    ###exponents
    elif type(expression) == exp:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol: 
            return expression * differentiate(inner_term)
        return expression
    
    ###secant
    elif type(expression) == sec:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return expression*tan(inner_term) * differentiate(inner_term)
        return expression*tan(inner_term)
    
    ###sin
    elif type(expression) == sin:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return cos(inner_term)* differentiate(inner_term)
        return cos(inner_term)
    
    ###cos
    elif type(expression) == cos:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return -sin(inner_term)* differentiate(inner_term)
        return -sin(inner_term)
    
    ###tan
    elif type(expression) == tan:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return (sec(inner_term)**2)* differentiate(inner_term)
        return sec(inner_term)**2
    
    ###cosecant
    elif type(expression) == csc:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return -expression*cot(inner_term) * differentiate(inner_term)
        return -expression*cot(inner_term)
    
    ###cotangent
    elif type(expression) == cot:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return (-csc(inner_term)**2) * differentiate(inner_term)
        return -csc(inner_term)**2
    
    ###arcsin
    elif type(expression) == asin:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return 1/((1-inner_term**2)**1/2) * differentiate(inner_term)
        return 1/((1-inner_term**2)**1/2)
    
    ###arccos
    elif type(expression) == acos:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return -1/((1-inner_term**2)**1/2) * differentiate(inner_term)
        return -1/((1-inner_term**2)**1/2)
    
    ###arctan
    elif type(expression) == atan:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return (-csc(inner_term)**2) * differentiate(inner_term)
        return 1/(1-inner_term**2)
    
    #if no matches, print notification string
    else:
        print(f'No matching differentiation method for {expression}')

def differentiate_withstep(expression, times = 1):
    for _ in range(times-1):
        expression = differentiate_withstep(expression)
    ###rule of constant
    # if expression is a constant
    if expression.is_number: 
        #print(f'd({expression}) = 0\n')
        return 0
    ###when expression is a symbol with a power of 1
    elif type(expression) == sym.Symbol:
        #print(f'd({expression}) = 1\n')
        return 1
    
    ###sum/difference rule
    elif type(expression) == sym.Add:
        res = 0
        for arg in expression.args:
            #print(arg,differentiate(arg))
            res += differentiate_withstep(arg)
        #print(res)
        print(f'{expression} is a sum, d({expression}) = {res}\n')
        return res
    
    ###multiple / product rule / quotient rule (when constant, one of the term will be 0)
    #sympy likes its multiples, so it expresses every thing in it
    if type(expression) == sym.Mul:
        #logarithms (with base)
        if type(expression.args[1]) == log:
            if hasattr(expression.args[0],'args'):
                if expression.args[0].args[0] == log:
                    base = expression.args[0].args[0].args[0]
                    inner_term = expression.args[1].args[0]
                    #chain rule
                    if type(inner_term) != sym.Symbol: 
                        print(f'{expression} is a non-natural based logarithm, d({expression}) = {1/(inner_term*ln(base))} * {differentiate_withstep(inner_term)}\nCHAINED: {inner_term}\n')
                        return 1/(inner_term*ln(base)) * differentiate_withstep(inner_term)
                    print(f'{expression} is a non-natural based logarithm, d({expression}) = {1/(inner_term*ln(base))}\n')
                    return 1/(inner_term*ln(base)) 
        
        #quotient
        if type(expression.args[0]) == sym.Pow and expression.args[0].args[1] == -1: #this is the way sympy represents fractions, with a power of -1
            f = expression.args[1]
            g = expression.args[0].args[0]
            print(f'{expression} is a quotient, d({expression}) = {(g*differentiate_withstep(f) - f*differentiate_withstep(g))/(g**2)}\n')
            return (g*differentiate_withstep(f) - f*differentiate_withstep(g))/(g**2)
        #product
        else:
            f = expression.args[1]
            g = expression.args[0]
            print(f'{expression} is a product, d({expression}) = {g*differentiate_withstep(f)+f*differentiate_withstep(g)}\n')
            return g*differentiate_withstep(f)+f*differentiate_withstep(g)
    
    ###power rule / exponent(non-natural)
    elif type(expression) == sym.Pow:
        power = expression.args[1]
        inner_term = expression.args[0]
        #print(f'power: {power}, innerterm: {inner_term}')
        ##exponent (non natrural based)
        if inner_term.is_number:
            inner_term = expression.args[1]
            base = expression.args[0]
            #chain rule
            if type(inner_term) != sym.Symbol:
                print(f'{expression} is a non natural exponent, d({expression}) = {expression*ln(base)} * {differentiate_withstep(inner_term)}\nCHAINED: {inner_term}\n')
                return expression*ln(base)*differentiate_withstep(inner_term)
            print(f'{expression} is a non natural exponent, d({expression}) = {expression*ln(base)}\n')
            return expression*ln(base)
        
        ##power
        else:
            #chain rule
            if type(inner_term) != sym.Symbol:
                print(f'{expression} is a power, d({expression}) = {(power*inner_term**(power-1))} * {differentiate_withstep(inner_term)}\nCHAINED: {inner_term}\n')
                return (power*inner_term**(power-1))*differentiate_withstep(inner_term)
            print(f'{expression} is an power, d({expression}) = {power*inner_term**(power-1)}\n')
            return power*inner_term**(power-1)
    
    ###logarithm (without base)
    if type(expression) == sym.log: #log = plain ln, or log base e
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol: 
            print(f'{expression} is a natural logarithm, d({expression}) = {(1/inner_term)} * {differentiate_withstep(inner_term)}\nCHAINED: {inner_term}\n')
            return (1/inner_term) * differentiate_withstep(inner_term)
        print(f'{expression} is a natural logarithm, d({expression}) = {(1/inner_term)}\n')
        return 1/inner_term
    
    ###exponents
    elif type(expression) == exp:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol: 
            print(f'{expression} is an exponent, d({expression}) = {(1/inner_term)} * {differentiate_withstep(inner_term)}\nCHAINED: {inner_term}\n')
            return expression * differentiate_withstep(inner_term)
        print(f'{expression} is an exponent, d({expression}) = {(1/inner_term)}\n')
        return expression
    
    ###secant
    elif type(expression) == sec:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            print(f'{expression} is a secant function, d({expression}) = {expression*tan(inner_term)} * {differentiate_withstep(inner_term)}\nCHAINED: {inner_term}\n')
            return expression*tan(inner_term) * differentiate_withstep(inner_term)
        print(f'{expression} is a secant function, d({expression}) = {expression*tan(inner_term)}\n')
        return expression*tan(inner_term)
    
    ###sin
    elif type(expression) == sin:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            print(f'{expression} is a sin function, d({expression}) = {cos(inner_term)} * {differentiate_withstep(inner_term)}\nCHAINED: {inner_term}\n')
            return cos(inner_term)* differentiate_withstep(inner_term)
        print(f'{expression} is a sin function, d({expression}) = {cos(inner_term)}\n')
        return cos(inner_term)
    
    ###cos
    elif type(expression) == cos:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return -sin(inner_term)* differentiate_withstep(inner_term)
        return -sin(inner_term)
    
    ###tan
    elif type(expression) == tan:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return (sec(inner_term)**2)* differentiate_withstep(inner_term)
        return sec(inner_term)**2
    
    ###cosecant
    elif type(expression) == csc:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return -expression*cot(inner_term) * differentiate_withstep(inner_term)
        return -expression*cot(inner_term)
    
    ###cotangent
    elif type(expression) == cot:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return (-csc(inner_term)**2) * differentiate_withstep(inner_term)
        return -csc(inner_term)**2
    
    ###arcsin
    elif type(expression) == asin:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return 1/((1-inner_term**2)**1/2) * differentiate_withstep(inner_term)
        return 1/((1-inner_term**2)**1/2)
    
    ###arccos
    elif type(expression) == acos:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return -1/((1-inner_term**2)**1/2) * differentiate_withstep(inner_term)
        return -1/((1-inner_term**2)**1/2)
    
    ###arctan
    elif type(expression) == atan:
        inner_term = expression.args[0]
        #chain rule
        if type(inner_term) != sym.Symbol:
            return (-csc(inner_term)**2) * differentiate_withstep(inner_term)
        return 1/(1-inner_term**2)
    
    #if no matches, print notification string
    else:
        print(f'No matching differentiation method for {expression}')


class Function():
    def __init__(self,function):
        self.function = function
        self.terms = self.function.args
        print(self.function, self.function.args, type(self.function))
        for term in self.terms:
            print(term, term.args, type(term))

#INVERSE RULE
# x = y+2y**2
# dy/dx = 1/(dx/dy)


#IMPLICITE DIFFERENTIATION

#Leibniz Rule

print()
print(differentiate_withstep((-x**3+9*x**2-16*x+1)*cos(2*x),6))
#a = Function(log(x,2))
#print(a.terms)
#print(differentiate_withstep(2*x**2))
#print(E**x)

#print(simplify(differentiate((x**2+1)*(x**3-1))))





