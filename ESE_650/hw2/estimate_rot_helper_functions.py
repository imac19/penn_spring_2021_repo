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

def get_q_delta(w, delta_t):
    q_delta = Quaternion()
    q_delta.from_axis_angle(w*delta_t)
    return q_delta


def process_model(prev_state, delta_t):
    q_delta = get_q_delta(prev_state[1], delta_t)
    q_return = q_delta * prev_state[0]
    return (q_return, prev_state[1])

def get_sigma_points_quaternion(P, Q, prev_state):
    prev_q = prev_state[0]
    prev_w = prev_state[1]
    n = Q.shape[1]
    W = list()
    S = np.linalg.cholesky(2*n*(P+Q))
    for i in range(0, n):
        W.append(S[:,i])
        W.append(-S[:,i])
    X = list()
    for col in W:
        q_w = Quaternion()
        q_w.from_axis_angle(col[0:3])
        w_w = col[3:6]
        q_add = q_w * prev_q
        w_add = prev_w + w_w
        to_add = get_state(q_add, w_add)
        X.append(to_add)
        
    return X

def get_transformed_sigmas(sigmas, delta_t):
    Y = list()
    for point in sigmas:
        Y.append(process_model(point,delta_t))
    return Y
    
def get_zrot(w):
    zrot = w 
    return zrot

def get_zacc(q):
    g = Quaternion(scalar=0, vec=[0,0,9.81])
    q_inv = q.inv()
    g_prime = q_inv * g * q
    zacc = g_prime.vec()
    return zacc

def get_projected_measurement_vectors(sigmas):
    Z = list()
    for point in sigmas:
        q = point[0]
        w = point[1]
        z_a = get_zacc(q)
        z_b = get_zrot(w)
        Z.append((z_a, z_b))
    
    return Z

def calculate_measurement_vector_mean(Z):
    elem_one = np.zeros(len(Z[0][0]))
    elem_two = np.zeros(len(Z[0][1]))
    count = 0
    
    for tup in Z:
        elem_one += np.array(tup[0])
        elem_two += np.array(tup[1])
        count += 1
    
    elem_one = elem_one/count
    elem_two = elem_two/count
    
    return (elem_one, elem_two)

def calculate_transformed_sigmas_mean(Y, prev_state_estimate):
    
    elem_two = np.zeros(len(prev_state_estimate[1]))
    count = 0
    
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
            e_i.normalize()
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
    
    w_bar = Y_mean[1]
    
    for point in Y:
#        point[0].normalize()
        q_bar = Y_mean[0]
        q_inv = q_bar.inv()
        q_i = point[0]
        rot_w = q_i * q_inv
        w_c = point[1] - w_bar
        rot_w.normalize()
        rot_c = rot_w.axis_angle()
        w_i = np.append(rot_c, w_c)
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
    
    z_mean = np.array([Z_mean[0], Z_mean[1]])
    
    for vec in Z:
        z_i = np.array([vec[0], vec[1]])
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
    
    w_bar = Y_mean[1]
    z_mean = np.array([Z_mean[0], Z_mean[1]])
    
    for i in range (0, len(Y)):
#        point[0].normalize()
        q_bar = Y_mean[0]
        q_inv = q_bar.inv()
        q_i = Y[i][0]
        rot_w = q_i * q_inv
        w_c = Y[i][1] - w_bar
        rot_w.normalize()
        rot_c = rot_w.axis_angle()
        w_i = np.append(rot_c, w_c)
        w_i = w_i.reshape((len(w_i), 1))
        
        z_i = np.array([Z[i][0], Z[i][1]])
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
    return P_x @ np.linalg.inv(P_v)

def state_update(x, K, v):
    K_array = np.array(K)
    v_array = np.array(v)

    
    updated = K_array @ v_array
    update = Quaternion()
    update.from_axis_angle(updated[0:3])
    update_q = update * x[0]
    to_return = (update_q, x[1]+updated[3:6])
    
    return to_return

def covariance_update(P_k, K, P_v):
    return P_k - (K @ P_v @ K.T)

def add_covariance_diagonal(m, val):
    n = len(m)
    for i in range(0, n):
        m[i][i] = val[i]
        
    return m

def accel_to_roll(accel_y, accel_z):
    r = []
    for i in range(0, len(accel_y)):
        r.append(math.atan2(accel_y[i], accel_z[i]))
        
    return r

def accel_to_pitch(accel_x, accel_y, accel_z):
    r = []
    for i in range(0, len(accel_x)):
        r.append(math.atan2(-accel_x[i], np.sqrt(accel_y[i]*accel_y[i] + accel_z[i]*accel_z[i])))
        
    return r



# =============================================================================
# def get_states(roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate):
#     states = []
#     for i in range(0, len(roll)):
#         q = Quaternion()
#         q.from_axis_angle([roll[i], pitch[i], yaw[i]])
#         w = (roll_rate[i], pitch_rate[i], yaw_rate[i])
#         
#         states.append(get_state(q, w))
#         
#     return states
# =============================================================================

# =============================================================================
# def get_states(roll_rate, pitch_rate, yaw_rate, timestamps):
#     states = []
#     for i in range(0, len(roll_rate)):
#         q = Quaternion()
#         q.from_axis_angle([roll_rate[i], pitch_rate[i], yaw_rate[i]])
#         w = (roll_rate[i], pitch_rate[i], yaw_rate[i])
#         
#         if (i>0):
#             q.q[0] = q.scalar()*(timestamps[0][i]-timestamps[0][i-1])
#         
#         states.append(get_state(q, w))
#         
#     return states
# =============================================================================


# =============================================================================
#     updated = x_array + (K_array @ v_array)
#     to_return = Quaternion()
#     to_return.from_axis_angle(updated[0:3])
#     to_return.normalize()
#     to_return = (to_return, updated[3:6])
# =============================================================================
# =============================================================================
# def get_q_noise(noise_vec):
#     q_noise = Quaternion()
#     q_noise.from_axis_angle(noise_vec)
#     return q_noise
# =============================================================================

# =============================================================================
# def get_q_disturbed(q_state, q_noise):
#     q_disturbed = q_state * q_noise
#     return q_disturbed
# 
# def get_w_disturbed(w, noise_w):
#     w_disturbed = w + noise_w
#     return w_disturbed 
# 
# def get_next_state(q_disturbed, q_delta, w_disturbed):
#     q_next = q_disturbed * q_delta
#     return (q_next, w_disturbed)
# 
# =============================================================================
# =============================================================================
# def process_model(state, delta_t, noise_vec = np.array([0,0,0]), noise_w=np.array([0,0,0])):
#     
# # =============================================================================
# #     noise_vec = np.random.normal(size=(1,3))
# #     noise_w = [.001,.001,.001]
# # =============================================================================
#     
#     q_state = state[0]
#     w_state = state[1]
#     q_delta = get_q_delta(w_state, delta_t)
#     q_noise = get_q_noise(noise_vec)
#     q_disturbed = get_q_disturbed(q_state, q_noise)
#     w_disturbed = get_w_disturbed(w_state, noise_w)
#     next_state = get_next_state(q_disturbed, q_delta, w_disturbed)
#     
#     return next_state
# =============================================================================

# =============================================================================
# def get_q_delta(w, delta_t):
# # =============================================================================
# #     q_delta = Quaternion()
# #     
# #     angle = (np.linalg.norm(w) * delta_t)
# #     if angle != 0:
# #         axis = w/np.linalg.norm(angle)
# #     else:
# #         axis = np.array([1,0,0])
# #     q_delta.q[0] = math.cos(angle/2)
# #     q_delta.q[1:4] = axis*math.sin(angle/2)
# #     q_delta.normalize()
# # =============================================================================
#     q_delta = Quaternion()
#     q_delta.from_axis_angle(w*delta_t)
#     return q_delta
# =============================================================================

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