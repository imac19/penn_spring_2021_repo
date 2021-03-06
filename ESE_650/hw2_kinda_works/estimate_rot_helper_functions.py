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
    y = np.cumsum(x)
    return y, x

def vicon_to_accel(orientations):
    yaws = orientations[0]
    rolls = orientations[1]
    pitches = orientations[2]
    accel_z = []
    accel_x = []
    accel_y = []
    g = [0,0,9.81]
    
    for i in range(0, len(yaws)):
        accel_z.append((yaws[i])*g[2])
        accel_x.append((rolls[i])*g[0])
        accel_y.append((pitches[i])*g[1])
        
    return [accel_z, accel_x, accel_y]

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
    q_delta = Quaternion()
    q_delta.from_axis_angle(w)
    return q_delta

def get_q_noise(noise_vec):
    q_noise = Quaternion()
    q_noise.from_axis_angle(noise_vec)
    return q_noise

def get_q_disturbed(q_state, q_noise):
    q_disturbed = q_state * q_noise 
    return q_disturbed

def get_w_disturbed(w, noise_w):
    w_disturbed = w + noise_w
    return w_disturbed 

def get_next_state(q_disturbed, q_delta, w_disturbed):
    q_next = q_disturbed * q_delta
    return (q_next, w_disturbed)

def process_model(state, noise_vec = (np.random.normal(size=(1,3)))/100, noise_w = [.00001,.00001,.00001]):
    
# =============================================================================
#     noise_vec = np.random.normal(size=(1,3))
#     noise_w = [.001,.001,.001]
# =============================================================================
    
    q_state = state[0]
    w_state = state[1]
    q_delta = get_q_delta(w_state)
    q_noise = get_q_noise(noise_vec)
    q_disturbed = get_q_disturbed(q_state, q_noise)
    w_disturbed = get_w_disturbed(w_state, noise_w)
    next_state = get_next_state(q_disturbed, q_delta, w_disturbed)
    
    return next_state

def get_zrot(w, vrot = np.random.normal(size=(1,3))):
    zrot = w + vrot
    return zrot

def get_zacc(q, vacc = np.random.normal(size=(1,3))):
    g = Quaternion(scalar=0, vec=[0,0,9.81])
    q_inv = q.inv()
    g_prime = q * g
    g_prime = g_prime * q_inv
    zacc = g_prime.vec() + vacc
    return zacc

# =============================================================================
# def get_sigma_points_normal(mean, covariance):
#     S = np.linalg.cholesky(covariance)
#     n = covariance.shape[1]
#     W = list()
#     for i in range(0,n):
#         col_vec_pos = (covariance[:,i]) * np.sqrt(2*n)
#         col_vec_neg = (covariance[:,i]) * np.sqrt(2*n)
#         W.append(col_vec_neg)
#         W.append(col_vec_pos)
#     
#     return W
# =============================================================================

def get_sigma_points_quaternion(P, Q, prev_state):
    prev_q = prev_state[0]
    prev_w = prev_state[1]
    n = Q.shape[1]
    W = list()
    cov = 2*n*(P+Q)
    S = np.linalg.cholesky(cov)
    for i in range(0, n):
        W.append(S[:,i])
        W.append(-S[:,i])
    X = list()
    for col in W:
        q_w = Quaternion()
        q_w.from_axis_angle(col[0:3])
        w_w = col[3:6]
        q_add = prev_q * q_w
        w_add = prev_w + w_w
        to_add = get_state(q_add, w_add)
        X.append(to_add)
        
    return X

def get_transformed_sigmas(sigmas):
    Y = list()
    for point in sigmas:
        Y.append(process_model(point,0,0))
    return Y
    
def get_projected_measurement_vectors(sigmas):
    Z = list()
    for point in sigmas:
        q = point[0]
        w = point[1]
        z_a = get_zacc(q, vacc=0)
        z_b = get_zrot(w, vrot=0)
        Z.append((z_a, z_b))
    
    return Z

def calculate_measurement_vector_mean(Z):
    for tup in Z:
        elem_one = np.zeros(len(tup[0]))
        elem_two = np.zeros(len(tup[1]))
        count = 0
        break
    
    for tup in Z:
        elem_one += np.array(tup[0])
        elem_two += np.array(tup[1])
        count += 1
    
    elem_one = elem_one/count
    elem_two = elem_two/count
    
    return (elem_one, elem_two)

def calculate_transformed_sigmas_mean(Y, prev_state_estimate):
    for tup in Y:
        elem_two = np.zeros(len(tup[1]))
        count = 0
        break
    
    for tup in Y:
        elem_two += np.array(tup[1])
        count += 1
    
    elem_two = elem_two/count
    
    q_bar = prev_state_estimate[0]
    iterations = 10
    for t in range(0, iterations):
        e_vec_list = []
        
        for tup in Y:
            q_i = tup[0]
            e_i = q_i * q_bar.inv()
            e_vec_list.append(e_i.axis_angle())
        
        count = 0
        sum_val = np.zeros(len(e_vec_list[0]))
        for e in e_vec_list:
            count += 1 
            add = np.array(e)
            sum_val += add
        
        e_estimate = Quaternion()
        e_estimate.from_axis_angle(sum_val/count)
        e_estimate.normalize()
        q_bar = e_estimate * q_bar
        
    return (q_bar, elem_two)

def calculate_transformed_sigmas_covariance(Y, Y_mean):
    n = len(Y_mean[0].q) + len(Y_mean[1]) - 1
    cov_matrix = np.zeros(shape=(n, n))
    count = 0
    
    rot_c_mean = Y_mean[0].axis_angle()
    w_c_mean = Y_mean[1]
    x_mean = np.append(rot_c_mean, w_c_mean)
    
    for point in Y:
        rot_c = point[0].axis_angle()
        w_c = point[1]
        x_i = np.append(rot_c, w_c)
        w_i = x_i - x_mean
        w_i = w_i.reshape((len(w_i), 1))
        
        cov_add = w_i @ w_i.T
        cov_matrix += cov_add
        count +=1
    
    cov_matrix = cov_matrix/count

    return cov_matrix

def calculate_predicted_measurements_covariance(Z, Z_mean):
    n = len(Z_mean[0]) + len(Z_mean[1]) 
    cov_matrix = np.zeros(shape=(n, n))
    count = 0
    
    z_mean = np.array(Z_mean)
    
    for vec in Z:
        z_i = np.array(vec)
        z_vec = z_i - z_mean
        z_vec = z_vec.reshape((n, 1))
        
        
        cov_add = z_vec @ z_vec.T
        cov_matrix += cov_add
        count +=1
    
    cov_matrix = cov_matrix/count
    
    return cov_matrix

def get_covariance_ukf(P, R):
    P_return = P + R
    return P_return

def get_cross_correlation(Y, Z, Y_mean, Z_mean):
    n = len(Y_mean[0].q) + len(Y_mean[1]) - 1
    cov_matrix = np.zeros(shape=(n, n))
    count = 0
    
    rot_c_mean = Y_mean[0].axis_angle()
    w_c_mean = Y_mean[1]
    x_mean = np.append(rot_c_mean, w_c_mean)
    
    z_mean = np.array(Z_mean)
    
    for i in range(0, len(Y)):
        rot_c = Y[i][0].axis_angle()
        w_c = Y[i][1]
        x_i = np.append(rot_c, w_c)
        w_i = x_i - x_mean
        w_i = w_i.reshape((len(w_i), 1))
        
        z_i = np.array(Z[i])
        z_vec = z_i - z_mean
        z_vec = z_vec.reshape((n, 1))
        
        
        cov_add = w_i @ z_vec.T
        cov_matrix += cov_add
        count +=1
    
    cov_matrix = cov_matrix/count
    
    return cov_matrix
    
def get_innovation(actual, predicted):
    return actual - predicted

def get_kalman_gain(P_x, P_v):
    return P_x * np.linalg.inv(P_v)

def state_update(x, K, v):
    x_array = np.append(x[0].axis_angle(), x[1])
    K_array = np.array(K)
    v_array = np.array(v)

    
    updated = x_array + (K_array @ v_array)
    to_return = Quaternion()
    to_return.from_axis_angle(updated[0:3])
    to_return.normalize()
    to_return = (to_return, updated[3:6])
    
    return to_return

def covariance_update(P_k, K, P_v):
    return P_k - (K*P_v*K.T)

def add_covariance_diagonal(m, val):
    n = len(m)
    for i in range(0, n):
        m[i][i] = val[i]
        
    return m

def get_states(roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate):
    states = []
    for i in range(0, len(roll)):
        q = Quaternion()
        q.from_axis_angle([roll[i], pitch[i], yaw[i]])
        w = (roll_rate[i], pitch_rate[i], yaw_rate[i])
        
        states.append(get_state(q, w))
        
    return states
