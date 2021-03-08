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
# =============================================================================
#     imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
#     vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
#     accel = imu['vals'][0:3,:]
#     gyro = imu['vals'][3:6,:]
#     T = np.shape(imu['ts'])[1]
#     vicon_rots = vicon['rots']
# =============================================================================
    
    imu = io.loadmat(os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat")) 
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

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
    accel_sensitivity = 32.6
    converted_accel_x = -accel_convert(accel[0], accel_biases[0], accel_sensitivity)
    converted_accel_y = -accel_convert(accel[1], accel_biases[1], accel_sensitivity)
    converted_accel_z = accel_convert(accel[2], accel_biases[2], accel_sensitivity)
    gyro_biases = [370,374,375.7]
    gyro_sensitivity = 115
    converted_gyro_yaw, converted_gyro_yaw_rate = gyro_convert(gyro[0], gyro_biases[0], gyro_sensitivity)
    converted_gyro_roll, converted_gyro_roll_rate = gyro_convert(gyro[1], gyro_biases[1], gyro_sensitivity)
    converted_gyro_pitch, converted_gyro_pitch_rate = gyro_convert(gyro[2], gyro_biases[2], gyro_sensitivity)
    converted_accel_roll = accel_to_roll(converted_accel_y, converted_accel_z)
    converted_accel_pitch = accel_to_pitch(converted_accel_x, converted_accel_y, converted_accel_z)
    
# =============================================================================
#     converted_vicon_rots = rot_matrix_convert(vicon_rots)
#     converted_vicon_accel = vicon_to_accel(converted_vicon_rots)
# =============================================================================
    
    
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
# =============================================================================
#     vicon_timestamps = vicon['ts']
# =============================================================================
    

# =============================================================================
#     plt.figure()
#     plt.plot(imu_timestamps[0], converted_gyro_yaw, label='IMU')
#     plt.plot(vicon_timestamps[0], converted_vicon_rots[0], label='Vicon')
#     plt.title('Converted Gyro Yaw vs Converted Vicon Yaw')
#     plt.legend()
#     plt.show()
#     
#     plt.figure()
#     plt.plot(imu_timestamps[0], converted_gyro_roll, label='IMU')
#     plt.plot(vicon_timestamps[0], converted_vicon_rots[1], label='Vicon')
#     plt.title('Converted Gyro Roll vs Converted Vicon Roll')
#     plt.legend()
#     plt.show()
#     
#     plt.figure()
#     plt.plot(imu_timestamps[0], converted_gyro_pitch, label='IMU')
#     plt.plot(vicon_timestamps[0], converted_vicon_rots[2], label='Vicon')
#     plt.title('Converted Gyro Pitch vs Converted Vicon Pitch')
#     plt.legend()
#     plt.show()
#     
#     plt.figure()
#     plt.plot(imu_timestamps[0], converted_accel_pitch, label='IMU')
#     plt.plot(vicon_timestamps[0], converted_vicon_rots[2], label='Vicon')
#     plt.title('Converted IMU Acceleration Pitch vs Converted Vicon Pitch')
#     plt.legend()
#     plt.show()
#     
#     plt.figure()
#     plt.plot(imu_timestamps[0], converted_accel_roll, label='IMU')
#     plt.plot(vicon_timestamps[0], converted_vicon_rots[1], label='Vicon')
#     plt.title('Converted IMU Acceleration Roll vs Converted Vicon Roll')
#     plt.legend()
#     plt.show()
# =============================================================================
    
# =============================================================================
#     print('Implementing UKF')
# =============================================================================
    
    P = np.zeros((6,6))
    Q = np.zeros((6,6))
    R = np.zeros((6,6))
    
    P = add_covariance_diagonal(P, [.1,.1,.1,.1,.1,.1])
    Q = add_covariance_diagonal(Q, [.0001,.0001,.0001,.0001,.0001,.0001])
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

    # Implement UKF
    
    updated_state_estimates = [(Quaternion(), [0, 0, 0])]
    updated_P_estimates = [P]
    prev_state = updated_state_estimates[0]
    prev_P = P
    
# =============================================================================
#     print('Starting State:')
#     print([prev_state[0].q, prev_state[1]])
#     print('Starting Covariance:')
#     print(prev_P)
# =============================================================================
    
    for i in range(1, len(imu_timestamps[0])):
# =============================================================================
#         print(i)
# =============================================================================
        z_k_m = np.array([converted_accel_x[i],
                        converted_accel_y[i], converted_accel_z[i],
                        converted_gyro_roll_rate[i], converted_gyro_pitch_rate[i], 
                        converted_gyro_yaw_rate[i]])
        
# =============================================================================
#         z_k_m = np.array([converted_gyro_roll_rate[i], converted_gyro_pitch_rate[i], 
#                         converted_gyro_yaw_rate[i], converted_accel_x[i],
#                         converted_accel_y[i], converted_accel_z[i]])
# =============================================================================
# =============================================================================
#         print(z_k_m)
# =============================================================================
        delta_t = imu_timestamps[0][i] - imu_timestamps[0][i-1]
# =============================================================================
#         delta_t = 1
# =============================================================================
        
        # Step 1,2 
        X = get_sigma_points_quaternion(prev_P, Q, prev_state)
# =============================================================================
#         print('Get quaternion sigma points successful:')
#         for point in X:
#             print([point[0].q, point[1]])
#         print()
# =============================================================================
        
        # Step 3
        Y = get_transformed_sigmas(X, delta_t)
# =============================================================================
#         print('Get transformed sigma points successful:')
#         for point in Y:
#             print([point[0].q, point[1]])
#         print()
# =============================================================================
        
        # Step 4
        x_hat_k = calculate_transformed_sigmas_mean(Y, prev_state)
        
        # Step 5, 6
        Pk = calculate_transformed_sigmas_covariance(Y, x_hat_k)
        
        # Step 7
        Z = get_projected_measurement_vectors(Y)
        
        # Step 8 
        z_k = calculate_measurement_vector_mean(Z)
        v = get_innovation(np.append(z_k[0], z_k[1]), z_k_m)
        
        # Step 9 
        Pzz = calculate_predicted_measurements_covariance(Z, z_k)
        Pvv = get_covariance_ukf(Pzz, R)
        
        # Step 10
        Pxz = get_cross_correlation(Y, Z, x_hat_k, z_k)
        K = get_kalman_gain(Pxz, Pvv)
        
        # Step 11
        updated_state = state_update(x_hat_k, K, v)
        updated_covariance = covariance_update(Pk, K, Pvv)
        
        # Set previous variable to current 
        prev_state = updated_state
        prev_P = updated_covariance
        
        # Log updated variables
        updated_state_estimates.append(updated_state)
        updated_P_estimates.append(updated_covariance)
        
# =============================================================================
#         print('Get projected measurement vectors successful:')
#         for point in Z:
#             print(point)
#         print()
# =============================================================================
        
# =============================================================================
#         print('Get means/covariances successful:')
#         print('Measurement Vector Mean:')
#         print(z_k)
#         print('Measurement Vector Covariance:')
#         print(Pzz)
#         print('Transformed Sigmas Mean:')
#         print([x_hat_k[0].q, x_hat_k[1]])
#         print('Transformed Sigmas Covariance:')
#         print(Pk)
#         print()
# =============================================================================

# =============================================================================
#         print('Get innovation successful:')
#         print(v)
#         print()
# =============================================================================
            
# =============================================================================
#         print('Get ukf covariance successful:')
#         print(Pvv)
#         print()
# =============================================================================

# =============================================================================
#         print('Get ukf cross correlation successful:')
#         print(Pxz)
#         print()
# =============================================================================

# =============================================================================
#         print('Get kalman gain successful:')
#         print(K)
#         print()
# =============================================================================
        
# =============================================================================
#         print('Get updated state successful:')
#         print([updated_state[0].q, updated_state[1]])
#         print()
# =============================================================================
        

# =============================================================================
#         print('Get updated covariance successful:')
#         print(updated_covariance)
#         print()
# =============================================================================
        
        
    
    roll = np.zeros(len(imu_timestamps[0]))
    pitch = np.zeros(len(imu_timestamps[0]))
    yaw = np.zeros(len(imu_timestamps[0]))
    for i in range(0, len(updated_state_estimates)):
        orientations = updated_state_estimates[i][0].euler_angles()
# =============================================================================
#         print(orientations)
# =============================================================================
        roll[i] = orientations[0]
        pitch[i] = orientations[1]
        yaw[i] = orientations[2]
        
            

    
    
# =============================================================================
#     plt.figure()
#     plt.plot(imu_timestamps[0], yaw, label='UKF')
#     plt.plot(vicon_timestamps[0], converted_vicon_rots[0], label='Vicon')
#     plt.title('UKF Yaw vs Converted Vicon Yaw')
#     plt.legend()
#     plt.show()
#     
#     plt.figure()
#     plt.plot(imu_timestamps[0], roll, label='UKF')
#     plt.plot(vicon_timestamps[0], converted_vicon_rots[1], label='Vicon')
#     plt.title('UKF Roll vs Converted Vicon Roll')
#     plt.legend()
#     plt.show()
#     
#     plt.figure()
#     plt.plot(imu_timestamps[0], pitch, label='UKF')
#     plt.plot(vicon_timestamps[0], converted_vicon_rots[2], label='Vicon')
#     plt.title('UKF Pitch vs Converted Vicon Pitch')
#     plt.legend()
#     plt.show()
# =============================================================================

    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw

# =============================================================================
# estimate_rot()
# =============================================================================
