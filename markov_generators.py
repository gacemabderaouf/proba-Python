import scipy.stats as st
import numpy as np

size = 800

def raw_input(s) :
    return input(s)

def uniform_generator(u) :
    a = int(raw_input("To generate uniform[a,b] :\nGive a : "))
    b = int(raw_input("Give b : "))
    
    data = (b-a) * u + a  
    print(data)
    return data