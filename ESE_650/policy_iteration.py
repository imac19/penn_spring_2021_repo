#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 21:44:53 2021

@author: ian
"""

import numpy as np

def get_grid_world(length, width):
    grid_world = np.zeros((length, width))
    return grid_world

print(get_grid_world(10,10))