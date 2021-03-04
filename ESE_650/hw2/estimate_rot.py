import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from estimate_rot_helper_functions import *
from matplotlib import pyplot as plt

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

    # your code goes here
    
    # check contents of data
    print('Checking the contents of the data')
    print()
    print('imu data')
    print(imu)
    print()
    print('vicon data')
    print(vicon)
    print()
    print('accel data')
    print(accel)
    print()
    print('gyro data')
    print(gyro)
    print()
    print('T data')
    print(T)
    print()
    print('Vicon rots')
    print(vicon_rots)
    print()
    print()
    
    # Convert accelerometer and gyroscope data
    
    accel_biases = [511,501,506]
    accel_sensitivity = 33
    converted_accel_x = -accel_convert(accel[0], accel_biases[0], accel_sensitivity)
    converted_accel_y = -accel_convert(accel[1], accel_biases[1], accel_sensitivity)
    converted_accel_z = accel_convert(accel[2], accel_biases[2], accel_sensitivity)
    gyro_biases = [370,374,376]
    gyro_sensitivity = 110
    converted_gyro_yaw = gyro_convert(gyro[0], gyro_biases[0], gyro_sensitivity)
    converted_gyro_roll = gyro_convert(gyro[1], gyro_biases[1], gyro_sensitivity)
    converted_gyro_pitch = gyro_convert(gyro[2], gyro_biases[2], gyro_sensitivity)
    converted_vicon_rots = rot_matrix_convert(vicon_rots)
    
    
    print('Converting data')
    print()
    print('converted accel data')
    print(converted_accel_x)
    print(converted_accel_y)
    print(converted_accel_z)
    print()
    print('converted gyro data')
    print(converted_gyro_yaw)
    print(converted_gyro_roll)
    print(converted_gyro_pitch)
    print()
    print('converted rot matrix data')
    print(converted_vicon_rots)
    print()
    print()
    quaternion = Quaternion()
    quaternion.from_rotm(vicon_rots[:,:,0])
    print(quaternion.euler_angles())
    
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
    plt.plot(imu_timestamps[0], converted_accel_x)
    plt.title('Converted Acceleration - X Axis')
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], converted_accel_y)
    plt.title('Converted Acceleration - Y Axis')
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], converted_accel_z)
    plt.title('Converted Acceleration - Z Axis')
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], accel[0])
    plt.title('Unconverted Acceleration - X Axis')
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], accel[1])
    plt.title('Unconverted Acceleration - Y Axis')
    plt.show()
    
    plt.figure()
    plt.plot(imu_timestamps[0], accel[2])
    plt.title('Unconverted Acceleration - Z Axis')
    plt.show()
    
    

    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw

estimate_rot()