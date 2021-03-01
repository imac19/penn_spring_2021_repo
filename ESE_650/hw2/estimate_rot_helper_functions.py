#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 19:18:48 2021

@author: ian
"""

import numpy as np
import math
from quaternion import Quaternion


def accel_convert(raw, bias, sensitivity):
    x = np.empty(len(raw))
    for i in range(0, len(raw)):
        x[i] = (raw[i]-bias) * (3300/(1023*sensitivity))
    return x
    
def gyro_convert(raw, bias, sensitivity):
    converted_sensitivity = sensitivity * 180
    x = np.empty(len(raw))
    for i in range(0, len(raw)):
        x[i] = (raw[i]-bias) * (3300/(1023*converted_sensitivity))
    x = np.cumsum(x)
    return x


def rot_matrix_convert(matrices):
    num_obs = matrices.shape[2]
    rolls = np.empty(num_obs)
    pitches = np.empty(num_obs)
    yaws = np.empty(num_obs)
    
    for i in range(0, num_obs):
        yaws[i] = math.atan2(matrices[1,0,i], matrices[0,0,i])
        pitches[i] = math.atan2((-matrices[2,0,i]), np.sqrt((matrices[2,1,i]**2) + (matrices[2,2,i]**2)))
        rolls[i] = math.atan2(matrices[2,1,i], matrices[2,2,i])
        
    return [yaws, rolls, pitches]

def get_state(q, w):
    return (q, w)

def get_q_delta(w):
    q_delta = Quaternion().from_axis_angle(w)
    return q_delta

def get_q_noise(noise_vec):
    q_noise = Quaternion().from_axis_angle(noise_vec)
    return q_noise

def get_q_disturbed(q_state, q_noise):
    q_disturbed = q_state.__mul__(q_noise)
    return q_disturbed

def get_w_disturbed(w, noise_w):
    w_disturbed = w + noise_w
    return w_disturbed 

def get_next_state(q_disturbed, q_delta, w_disturbed):
    q_next = q_disturbed.__mul__(q_delta)
    return (q_next, w_disturbed)

def process_model(state):
    
    noise_vec = np.random.normal(size=(1,3))
    noise_w = [.001,.001,.001]
    
    q_state = state[0]
    w_state = state[1]
    q_delta = get_q_delta(w_state)
    q_noise = get_q_noise(noise_vec)
    q_noise = get_q_noise(noise_vec)
    q_disturbed = get_q_disturbed(q_state, q_noise)
    w_disturbed = get_w_disturbed(w_state, noise_w)
    next_state = get_next_state(q_disturbed, q_delta, w_disturbed)
    
    return next_state

def get_zrot(w):
    vrot = np.random.normal(size=(1,3))
    zrot = w + vrot
    return zrot

def get_zacc(q):
    g = Quaternion(scalar=0, vector=[0,0,-9.81])
    q_inv = q.inv()
    g_prime = q.__mul__(g)
    g_prime = g_prime.__mul__(q_inv)
    vacc = np.random.normal(size=(1,3))
    zacc = g_prime.vec() + vacc
    return zacc

def get_sigma_points(mean, covariance):
    S = np.linalg.cholesky(covariance)
    n = covariance.shape[1]
    W = set()
    for i in range(0,n):
        col_vec_pos = (covariance[:,i]) * np.sqrt(2*n)
        col_vec_neg = (covariance[:,i]) * np.sqrt(2*n)
        W.add(col_vec_neg)
        W.add(col_vec_pos)
    
        