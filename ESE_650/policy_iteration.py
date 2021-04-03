#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 21:44:53 2021

@author: ian
"""

import numpy as np
from matplotlib import pyplot as plt
from seaborn import heatmap

def get_grid_world(length, width):
    grid_world = np.zeros((length, width))
    return grid_world

def get_grid_world_with_obstacles(length, width):
    grid_world = np.zeros((length, width))
    for i in range(0, length):
        for j in range(0, width):
            if i==0 or i==length-1 or j==0 or j==width-1:
                grid_world[i][j]=-10
                
    grid_world[5][4]=-10
    grid_world[4][4]=-10
    grid_world[3][4]=-10
    grid_world[2][4]=-10
    grid_world[2][5]=-10
    
    grid_world[7][3]=-10
    grid_world[7][4]=-10
    grid_world[7][5]=-10
    grid_world[7][6]=-10
    
    grid_world[4][7]=-10
    grid_world[5][7]=-10
    
    grid_world[8][8]=10
    
    return grid_world

def print_grid(grid):
    plt.figure()
    heatmap(grid, xticklabels=range(0,10), yticklabels=range(0,10), linewidths=.1,
            linecolor='black', square=True, annot=True)
    plt.show()
    
# =============================================================================
# Actions:
# (-1,0) = Move North
# (0, 1) = Move East
# (1, 0) = Move South
# (0,-1) = Move West
# =============================================================================
def get_transition_probabilities_and_states(action, state):
    if state==(0,0):
        if action==(-1, 0):
            return {(0,1):.5, state:.5}
        elif action==(0,-1):
            return {(1,0):.5, state:.5}
    elif state==(0,9):
        if action==(0,1):
            return {(1,9):.5, state:.5}
        elif action==(-1,0):
            return {(0,8):.5, state:.5}
    elif state==(9,0):
        if action==(0,-1):
            return {(8,0):.5, state:.5}
        elif action==(1,0):
            return {(9,1):.5, state:.5}
    elif state==(9,9):
        if action==(1,0):
            return {(9,8):.5, state:.5}
        elif action==(0,1):
            return {(8,9):.5, state:.5}
    
    if state[0]==0 and action==(-1,0):
        return {(state[0],state[1]+1):.333, (state[0],state[1]-1):.333, state:.333}
    elif state[1]==9 and action==(0,1):
        return {(state[0]-1,state[1]):.333, (state[0]+1,state[1]):.333, state:.333}
    elif state[0]==9 and action==(1,0):
        return {(state[0],state[1]+1):.333, (state[0],state[1]-1):.333, state:.333}
    elif state[1]==0 and action==(0,-1):
        return {(state[0]-1,state[1]):.333, (state[0]+1,state[1]):.333, state:.333}
    
    if state[0]==0:
        if action==(0,-1):
            return {(state[0], state[1]-1):.8, (state[0]+1, state[1]):.1, state:.1}
        elif action==(0,1):
            return {(state[0], state[1]+1):.8, (state[0]+1, state[1]):.1, state:.1}
    if state[1]==9:
        if action==(-1,0):
            return {(state[0]-1, state[1]):.8, (state[0], state[1]-1):.1, state:.1}
        elif action==(1,0):
            return {(state[0]-1, state[1]):.8, (state[0], state[1]-1):.1, state:.1}
    if state[0]==9:
        if action==(0,-1):
            return {(state[0], state[1]-1):.8, (state[0]-1, state[1]):.1, state:.1}
        elif action==(0,1):
            return {(state[0], state[1]+1):.8, (state[0]-1, state[1]):.1, state:.1}
    if state[1]==0:
        if action==(-1,0):
            return {(state[0]-1, state[1]):.8, (state[0], state[1]+1):.1, state:.1}
        elif action==(1,0):
            return {(state[0]-1, state[1]):.8, (state[0], state[1]+1):.1, state:.1}
    
    if action==(-1,0):
        return {(state[0]-1, state[1]):.7, (state[0], state[1]+1):.1, (state[0], state[1]+1):.1, state:.1}
    elif action==(1,0):
        return {(state[0]+1, state[1]):.7, (state[0], state[1]+1):.1, (state[0], state[1]+1):.1, state:.1}
    elif action==(0,1):
        return {(state[0], state[1]+1):.7, (state[0]-1, state[1]):.1, (state[0]+1, state[1]):.1, state:.1}
    elif action==(0,-1):
        return {(state[0], state[1]-1):.7, (state[0]-1, state[1]):.1, (state[0]+1, state[1]):.1, state:.1}
    
    else:
        return {state:1}
    
    
def initialize_policy(length, width):
    policy = np.empty((length, width), dtype='object')
    for i in range(len(policy)):
        for j in range(len(policy[0])):
            policy[i][j]=(0,1)
    return policy
    

def update_value_map(value_map, terminal_cost_grid, discount):
    updated_value_map = np.empty((10,10))
    for i in range(0,10):
        for j in range(0,10):
            current_state = (i,j)
            possible_actions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]
            resultant_values = np.zeros(5)
            
            for k, action in enumerate(possible_actions):
                t = get_transition_probabilities_and_states(action, current_state)
                v = terminal_cost_grid[current_state]
                
                for new_state in t.keys():
                    v+= (discount * t[new_state] * value_map[new_state])
                
                resultant_values[k] = v
                
            if np.argmax(resultant_values)!=0:
                v+=1
                
            updated_value_map[current_state] = max(resultant_values)
            
    return updated_value_map
    
    
# =============================================================================
# grid_world_with_obstacles = get_grid_world_with_obstacles(10,10)
# grid_world = get_grid_world(10, 10)
# print(grid_world)
# print(grid_world_with_obstacles)
# print_grid(grid_world)
# print_grid(grid_world_with_obstacles)
# =============================================================================
# =============================================================================
# test_state = (0,3)
# test_action = (0,1)
# test_transitions = get_transition_probabilities_and_states(test_action, test_state)
# print(test_transitions)
# =============================================================================
initial_policy = initialize_policy(10,10)
terminal_cost_grid = get_grid_world_with_obstacles(10, 10)
value_map = get_grid_world(10, 10)
discount=.9
print(initial_policy)
print(value_map)

iterations=16
for i in range (iterations):
    value_map = np.round(update_value_map(value_map, terminal_cost_grid, discount), 2)
    print(value_map)
    
print_grid(value_map)