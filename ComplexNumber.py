# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 10:45:55 2020

@author: lefin
"""
from sympy import symbols, Symbol, Rational, sqrt, solve, cos, tan, sin, atan, pi


i = Symbol('i')
Re,Im = symbols('Re,Im')


def radPi(flt):
    return Rational(flt/pi)

class Z():
    def __init__(self,real,imaginary):
        self.real = real
        self.imaginary = imaginary
        self.r = sqrt(self.real**2+self.imaginary**2)
        if real != 0:
            self.arg = atan(Rational(imaginary/real))
        
    def conjugate(self):#returns conjugate
        return Z(self.real, self.imaginary*-1)
    
    def con(self):#returns conjugate (shortened)
        return Z(self.real, self.imaginary*-1)
    
    def modulus(self):
        return sqrt(self.real**2+self.imaginary**2)
    
    def mod(self):
        return sqrt(self.real**2+self.imaginary**2)
    
    def pol(self):
        if self.arg >=0:
            return f'{self.r}(cos({radPi(self.arg)}*𝜋) + isin({radPi(self.arg)}*𝜋))'
        else:
            return f'{self.r}(cos({radPi(self.arg)}*𝜋) - isin({radPi(self.arg*-1)}*𝜋))'
    
    def exp(self):
        if self.arg >=0:
            return f'{self.r}e^(i{radPi(self.arg)}*𝜋)'
        else:
            return f'{self.r}e^(-i{radPi(self.arg*-1)}*𝜋)'
    
    def __str__(self):
        if self.imaginary >= 0:
            return f'{self.real} + {self.imaginary}i'
        else:
            return f'{self.real} - {-1 * self.imaginary}i'
    
    def __add__(self,other):
        if type(other) == Z or issubclass(type(other), Z):
            return Z(self.real + other.real, self.imaginary + other.imaginary)
        else:
            return Z(self.real + other, self.imaginary)
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
    def __neg__(self):
        return Z(-1 * self.real, -1*self.imaginary)
    
    def __mul__(self,other):
        if type(other) == Z or issubclass(type(other), Z):
            arg = self.arg + other.arg
            r = self.r * other.r
            return Z(r*cos(arg),r*sin(arg))
        else:
            return Z(self.real * other, self.imaginary * other)
    def __truediv__(self,other):
        if type(other) == Z or issubclass(type(other), Z):
            arg = self.arg - other.arg
            r = self.r / other.r
            return Z(r*cos(arg),r*sin(arg))
        else:
            return Z(self.real / other, self.imaginary / other)
        
class ZP(Z):
    def __init__(self,r,arg):
        super().__init__(r*cos(arg),r*sin(arg))


class i(Z):
    def __init__(self,imaginary):
        super().__init__(0,imaginary)
    def __str__(self):
        return f'{self.imaginary}i'
    

    
print(Z(1,1)*Z(1,1),-Z(2,-1)+Rational(1,2)+i(20))

print(ZP(sqrt(2),pi/4))




        


    