# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:20:04 2021

@author: lefin
"""

from fractions import Fraction

BASE_NUM_DICT = {'0': 0,
                 '1': 1,
                 '2': 2,
                 '3': 3,
                 '4': 4,
                 '5': 5,
                 '6': 6,
                 '7': 7,
                 '8': 8,
                 '9': 9,
                 'A': 10,
                 'B': 11,
                 'C': 12,
                 'D': 13,
                 'E': 14,
                 'F': 15,
                 'G': 16,
                 'H': 17,
                 'I': 18,
                 'J': 19,
                 'K': 20,
                 'L': 21,
                 'M': 22,
                 'N': 23,
                 'O': 24,
                 'P': 25,
                 'Q': 26,
                 'R': 27,
                 'S': 28,
                 'T': 29,
                 'U': 30,
                 'V': 31,
                 'W': 32,
                 'X': 33,
                 'Y': 34,
                 'Z': 35,
                 '.': '.'}

INV_BASE_NUM_DICT = {v: k for k, v in BASE_NUM_DICT.items()}


    
def base2dec(val: str,base: int, space = 2, display = True, fraction = False):
    is_f = False
    if '.' in val:
        is_f = True
    #calculations
    ival_ls = val.split('.')#seperate digits
    try:
        int_ls = [BASE_NUM_DICT[digit] for digit in list(ival_ls[0])] #convert digits to int
        if is_f:
            f_ls = [BASE_NUM_DICT[digit] for digit in list(ival_ls[1])]
    except:
        print('Digit in val unreconized.')
        return
    
    plc_ls = [place_val * base ** place for place,place_val in enumerate(int_ls[::-1])][::-1] #convert digits to decimal equivilant values
    if is_f:
        
        f_plc_ls = [place_val * Fraction(1/base) ** place for place,place_val in enumerate(f_ls,1)][::-1] #convert digits to decimal equivilant values
        res = sum(plc_ls+f_plc_ls) #result
    else:
        res = sum(plc_ls) #result
    
    if display:
        print(f'\n  BASE {base} to DECIMAL CONVERSION')
        print(f'=================================')
        print(str(val) + '\n')
        #display process
        places = [f'{base}^{place}' for place, _ in enumerate(list(ival_ls)[0])][::-1]
        if is_f:
            f_places = [f'{base}^-{place}' for place, _ in enumerate(list(ival_ls)[1],1)]
        display = zip(places,list(ival_ls)[0],map(str,int_ls),map(str,plc_ls))
        if is_f:
            display = zip(places+['.']+f_places,list(ival_ls[0])+['.']+list(ival_ls[1]),map(str,int_ls+['.']+f_ls),map(str,plc_ls+[' ']+f_plc_ls))
        display_ls = [[],[],[],[]]
        for column in display:
            max_len = len(max(column,key=len)) + space
            for vidx,val in enumerate(column):
                display_ls[vidx].append(val + ' '*(max_len-len(val)))
        
        
        print('PLACES ||   '+''.join(display_ls[0]) + '\n')
        print('ORIGIN ||   '+''.join(display_ls[1]) + '\n')
        print('INTVAL ||   '+''.join(display_ls[2]) + '\n')
        print('RELVAL ||   '+''.join(display_ls[3]) + '\n')
        
        print('\nRESULT ||   '+str(res))
    
    if fraction:
        return res
    else:
        return float(res)

def dec2base(val: int, base: int, space = 2, display = True, thresh = 6):
    is_f = False
    oval = val
    if type(val) == float:
        is_f = True
        thresh += 1
        val_f = val - int(val) #float part of the val is val - int(val)
        def f_conversion(value, place):
            if display:
                print(f'{base}^-{place}: {value}*{base} -> {value*base}')
            if place >= thresh:
                return [int(value)]
            else:
                value *= base
                return [int(value)] + f_conversion(value-int(value),place+1)
        val = int(val)
        
    
    #calculations
    def conversion(value, place):
        if display:
            
            print(f'{base}^{place}: {base}|{value} ... {value%base}')
        if value == 1:
            return [value]
        elif value < 1:
            return []
        else:
            #keep the remainder, then convert the quotient(kinda, idk)
            return [value%base] + conversion(value//base,place+1)
    
    if display:
        print(f'\nDECIMAL to BASE {base} CONVERSION')
        print(f'=================================')
        print(str(oval) + '\n')
        print('SHORT DIVISION PROCESS: \n')
    
    int_ls = conversion(val,0)[::-1]
    if int_ls == []:
        int_ls.append(0)
    if is_f:
        if display:
            print('FLOATING POINTS MULTIPLICATION PROCESS: \n')
        int_ls += ['.'] + f_conversion(val_f, 1)
    val_ls = [INV_BASE_NUM_DICT[inte] for inte in int_ls]
    res = ''.join(val_ls)
    
    
    if display:
        print('\nRESULT ||   '+str(res))
    
    return res

def base2base(val:str,vbase:int,dbase:int,space = 2, display = True, thresh = 6):
    return dec2base(base2dec(val,vbase,space,display),dbase,space,display,thresh)

base2base('0110100', 2, 10, thresh = 10)

#dec2base('1.078125',4, thresh = 50)
