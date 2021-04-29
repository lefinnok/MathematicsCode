# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:32:29 2021

@author: lefin
"""
import numpy as np
from math import ceil



Truth_Table = 'ABCS00000011010101101001101011001111'
syms = [char for char in Truth_Table if not char.isnumeric()]
TTlist = np.array(list(Truth_Table)).reshape(int(len(Truth_Table)/len(syms)),len(syms)).T
TTTable = {}

for var_row in TTlist:
    TTTable[var_row[0]] = list(map(int,var_row[1:]))
    print(f'{var_row[0]}: {list(map(int,var_row[1:]))}')




#kmap = np.zeros(((len(syms)-1)//2 * 2,ceil((len(syms)-1)/2)*2))


