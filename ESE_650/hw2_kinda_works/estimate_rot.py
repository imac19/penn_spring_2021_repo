import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from estimate_rot_helper_functions import *
from matplotlib import pyplot as plt
import os

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]
    vicon_rots = vicon['rots']
    
# =============================================================================
#     imu = io.loadmat(os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat")) 
#     accel = imu['vals'][0:3,:]
#     gyro = imu['vals'][3:6,:]
#     T = np.shape(imu['ts'])[1]
# =============================================================================

    # your code goes here
    
    # check contents of data
# =============================================================================
#     print('Checking the contents of the data')
#     print()
#     print('imu data')
#     print(imu)
#     print()
#     print('vicon data')
#     print(vicon)
#     print()
#     print('accel data')
#     print(accel)
#     print()
#     print('gyro data')
#     print(gyro)
#     print()
#     print('T data')
#     print(T)
#     print()
#     print('Vicon rots')
#     print(vicon_rots)
#     print()
#     print()
# =============================================================================
    
    # Convert accelerometer and gyroscope data
    
    accel_biases = [511,501,506]
    accel_sensitivity = 33
    converted_accel_x = -accel_convert(accel[0], accel_biases[0], accel_sensitivity)
    converted_accel_y = -accel_convert(accel[1], accel_biases[1], accel_sensitivity)
    converted_accel_z = accel_convert(accel[2], accel_biases[2], accel_sensitivity)
    gyro_biases = [370,374,376]
    gyro_sensitivity = 110
    converted_gyro_yaw, converted_gyro_yaw_rate = gyro_convert(gyro[0], gyro_biases[0], gyro_sensitivity)
    converted_gyro_roll, converted_gyro_roll_rate = gyro_convert(gyro[1], gyro_biases[1], gyro_sensitivity)
    converted_gyro_pitch, converted_gyro_pitch_rate = gyro_convert(gyro[2], gyro_biases[2], gyro_sensitivity)
    
    
    converted_vicon_rots = rot_matrix_convert(vicon_rots)
    converted_vicon_accel = vicon_to_accel(converted_vicon_rots)
    
    
# =============================================================================
#     print('Converting data')
#     print()
#     print('converted accel data')
#     print(converted_accel_x)
#     print(converted_accel_y)
#     print(converted_accel_z)
#     print()
#     print('converted gyro data')
#     print(converted_gyro_yaw)
#     print(converted_gyro_roll)
#     print(converted_gyro_pitch)
#     print()
#     print('converted rot matrix data')
#     print(converted_vicon_rots)
#     print()
#     print()
# =============================================================================
    
    # Plot converted data
    imu_timestamps = imu['ts']
    vicon_timestamps = vicon['ts']
    

    plt.figure()
    plt.plot(imu_timestamps[0], converted_gyro_yaw, label='IMU')
    plt.plot(vicon_timestamps[0], converted_vicon_rots[0], label='Vicon')
    plt.title('Converted Gyro Yaw vs Converted Vicon Yaw')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], converted_gyro_roll, label='IMU')
    plt.plot(vicon_timestamps[0], converted_vicon_rots[1], label='Vicon')
    plt.title('Converted Gyro Roll vs Converted Vicon Roll')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], converted_gyro_pitch, label='IMU')
    plt.plot(vicon_timestamps[0], converted_vicon_rots[2], label='Vicon')
    plt.title('Converted Gyro Pitch vs Converted Vicon Pitch')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], converted_accel_x, label='IMU')
    plt.plot(vicon_timestamps[0], converted_vicon_accel[1], label='Vicon')
    plt.title('Converted IMU Acceleration vs Converted Vicon Acceleration - X Axis')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], converted_accel_y, label='IMU')
    plt.plot(vicon_timestamps[0], converted_vicon_accel[2], label='Vicon')
    plt.title('Converted IMU Acceleration vs Converted Vicon Acceleration - Y Axis')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], converted_accel_z, label='IMU')
    plt.plot(vicon_timestamps[0], converted_vicon_accel[0], label='Vicon')
    plt.title('Converted IMU Acceleration vs Converted Vicon Acceleration - Z Axis')
    plt.legend()
    plt.show()
    
# =============================================================================
#     print('Implementing UKF')
# =============================================================================
    
    P = np.zeros((6,6))
    Q = np.zeros((6,6))
    R = np.zeros((6,6))
    
    P = add_covariance_diagonal(P, [.01,.01,.01,.01,.01,.01])
    Q = add_covariance_diagonal(Q, [0.0001, .0001, .0001, .001, .001, .001])
    R = add_covariance_diagonal(R, [10,10,10,10,10,10])
    
    
    
# =============================================================================
#     print('Initial P (State Error Covariance):')
#     print(P)
#     print()
#     
#     print('Initial Q (Dynamics Noise Covariance):')
#     print(Q)
#     print()
#     
#     print('Initial R (Measurement Noise Covariance):')
#     print(R)
#     print()
#     
#     print('Get Previous State Estimations:')
# =============================================================================
    estimated_states = get_states(converted_gyro_roll, converted_gyro_pitch, 
                        converted_gyro_yaw, converted_gyro_roll_rate, 
                        converted_gyro_pitch_rate, converted_gyro_yaw_rate)
# =============================================================================
#     print(estimated_states[0:5])
#     print()
# =============================================================================
    
    # Implement UKF
    
    updated_state_estimates = [estimated_states[0]]
    updated_P_estimates = [P]
    prev_state = estimated_states[0]
    prev_P = P
    
    for i in range(1, len(estimated_states)):
# =============================================================================
#         print(i)
# =============================================================================
        z_k = np.array([converted_gyro_roll[i], converted_gyro_pitch[i], 
                        converted_gyro_yaw[i], converted_gyro_roll_rate[i], 
                        converted_gyro_pitch_rate[i], converted_gyro_yaw_rate[i]])
        
        # Get quaternion sigma points
        quaternion_sigma_points = get_sigma_points_quaternion(P, Q, prev_state)
# =============================================================================
#         print('Get quaternion sigma points successful:')
#         print(quaternion_sigma_points)
#         print()
# =============================================================================
        
        # Get transformed sigma points
        transformed_sigma_points = get_transformed_sigmas(quaternion_sigma_points)
# =============================================================================
#         print('Get transformed sigma points successful:')
#         print(transformed_sigma_points)
#         print()
# =============================================================================
        
        # Get projected measurement vectors
        projected_measurement_vectors = get_projected_measurement_vectors(quaternion_sigma_points)
# =============================================================================
#         print('Get projected measurement vectors successful:')
#         print(projected_measurement_vectors)
#         print()
# =============================================================================
        
        # Get means/covariances
        measurement_vector_mean = calculate_measurement_vector_mean(projected_measurement_vectors)
        measurement_vector_covariance = calculate_predicted_measurements_covariance(
            projected_measurement_vectors, measurement_vector_mean)
        transformed_sigmas_mean = calculate_transformed_sigmas_mean(transformed_sigma_points, prev_state)
        transformed_sigmas_covariance = calculate_transformed_sigmas_covariance(
            transformed_sigma_points, transformed_sigmas_mean)
# =============================================================================
#         print('Get means/covariances successful:')
#         print(measurement_vector_mean)
#         print(measurement_vector_covariance)
#         print(transformed_sigmas_mean)
#         print(transformed_sigmas_covariance)
#         print()
# =============================================================================
        
        # Get innovation
        innovation = get_innovation(
            np.append(measurement_vector_mean[0], measurement_vector_mean[1]),
                     z_k)
# =============================================================================
#         print('Get innovation successful:')
#         print(innovation)
#         print()
# =============================================================================
            
        # Get UKF covariance
        ukf_covariance = get_covariance_ukf(measurement_vector_covariance, R)
# =============================================================================
#         print('Get ukf covariance successful:')
#         print(ukf_covariance)
#         print()
# =============================================================================
        
        # Get UKF cross correlation
        ukf_cross_correlation = get_cross_correlation(
            transformed_sigma_points, projected_measurement_vectors, 
            transformed_sigmas_mean, measurement_vector_mean)
# =============================================================================
#         print('Get ukf cross correlation successful:')
#         print(ukf_cross_correlation)
#         print()
# =============================================================================
        
        # Get kalman gain
        kalman_gain = get_kalman_gain(ukf_cross_correlation, ukf_covariance)
# =============================================================================
#         print('Get kalman gain successful:')
#         print(kalman_gain)
#         print()
# =============================================================================
        
        # Update state
        updated_state = state_update(transformed_sigmas_mean, kalman_gain, innovation)
# =============================================================================
#         print('Get updated state successful:')
#         print(updated_state)
#         print()
# =============================================================================
        
        # Update covariance 
        updated_covariance = covariance_update(transformed_sigmas_covariance, kalman_gain, ukf_covariance)
# =============================================================================
#         print('Get updated covariance successful:')
#         print(ukf_covariance)
#         print()
# =============================================================================
        
        # Set previous variable to current 
        prev_state = updated_state
        prev_P = updated_covariance
        
        # Log updated variables
        updated_state_estimates.append(updated_state)
        updated_P_estimates.append(updated_covariance)
        
    
    roll = np.zeros(len(estimated_states))
    pitch = np.zeros(len(estimated_states))
    yaw = np.zeros(len(estimated_states))
    for i in range(0, len(updated_state_estimates)):
        orientations = updated_state_estimates[i][0].euler_angles()
        roll[i] = orientations[0]
        pitch[i] = orientations[1]
        yaw[i] = orientations[2]
# =============================================================================
#     roll = converted_gyro_roll
#     pitch = converted_gyro_pitch
#     yaw = converted_gyro_yaw
# =============================================================================
    
    plt.figure()
    plt.plot(imu_timestamps[0], yaw, label='UKF')
    plt.plot(vicon_timestamps[0], converted_vicon_rots[0], label='Vicon')
    plt.title('UKF Yaw vs Converted Vicon Yaw')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], roll, label='UKF')
    plt.plot(vicon_timestamps[0], converted_vicon_rots[1], label='Vicon')
    plt.title('UKF Roll vs Converted Vicon Roll')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], pitch, label='UKF')
    plt.plot(vicon_timestamps[0], converted_vicon_rots[2], label='Vicon')
    plt.title('UKF vs Converted Vicon Pitch')
    plt.legend()
    plt.show()

    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw

estimate_rot()
