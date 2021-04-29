# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:42:21 2021

@author: lefin
"""
from BaseConversion import dec2base, base2dec
from BooleanAlgebra import binary_range


def ALU(A,B,Carry):
    out = 0
    car = 0
    if sum([A,B,Carry]) >= 1:
        out = 1
    if sum([A,B,Carry]) >= 2:
        car = 1
    if sum([A,B,Carry]) == 2:
        out = 0
    return car,out

def bi_sum(val_a, val_b):
    a_ls = list(val_a)
    b_ls = list(val_b)
    cal_gen = zip(a_ls[::-1],b_ls[::-1])
    res_ls = [(0,0)]
    for A,B in cal_gen:
        res_ls.append(ALU(int(A),int(B),res_ls[-1][0]))
    res_ls.pop(0)
    res = ''.join([str(n[1]) for n in res_ls][::-1])
    return res

def bi_sum_bit(val_a, val_b,bit):
    a_ls = ['0']*(bit-len(val_a)) + list(val_a)
    b_ls = ['0']*(bit-len(val_b)) + list(val_b)
    cal_gen = zip(a_ls[::-1],b_ls[::-1])
    res_ls = [(0,0)]
    for A,B in cal_gen:
        res_ls.append(ALU(int(A),int(B),res_ls[-1][0]))
    res_ls.pop(0)
    res = ''.join([str(n[1]) for n in res_ls][::-1])
    return res

def bit_format(val,bit):
    return ''.join(['0']*(bit-len(val)) + list(val))

def int_to_2_complement_word(val:int):
    bi = dec2base(val,2,display=False)
    bi = ('0'*max(8-len(bi),0))+bi
    print(bi)
    one_complement = ''.join(map(str,[int(not int(v)) for v in list(bi)]))
    print(one_complement)
    two_complement = bi_sum(one_complement,'00000001')
    print(two_complement)
    return two_complement

for x in binary_range(5):
    print(''.join(x))


'''
for n in range(8):
    print(bit_format(dec2base(n,2,display=False),4), bi_sum_bit(dec2base(n,2,display=False),dec2base(2,2,display=False),4))
'''


