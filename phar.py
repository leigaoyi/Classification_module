# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:48:17 2020

@author: kasy
"""

from time import sleep
from tqdm import trange, tqdm
from multiprocessing import Pool, freeze_support

L = list(range(9))

def progresser(n):
    interval = 0.001 / (n + 2)
    total = 5000
    text = "#{0}, est. {1:<04.2}s".format(n, interval * total)
    for _ in trange(total, desc=text, position=n):
        sleep(interval)
        
    