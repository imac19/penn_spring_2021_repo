import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import hw0_key
import hw0_solution
from autograder import Autograder

def test_compute_derivative(cut, key, autograder):
    #1D test
    x = np.linspace(0, 2*np.pi, 10000000)
    sine = np.sin(x)
    key_output = key.compute_derivative(2*np.pi/10000000, sine)

    #define tests
    test_id_type = autograder.create_test('Problem 2, 1D Output Type', 2)
    test_id_shape = autograder.create_test('Problem 2, 1D Output Shape', 2)
    test_id_val = autograder.create_test('Problem 2, 1D Correct Output Values (RMSE)', 4, 0.01, 10)
    test_id_time = autograder.create_test('problem 2, 1D time (sec)', 2, 1, 10)
    try:
        time_start = time.perf_counter()
        test_output = cut.compute_derivative(2*np.pi/10000000, sine)
        time_elapsed = time.perf_counter() - time_start
        print(time_elapsed)

        if type(test_output) == torch.Tensor:
            autograder.score_test(test_id_type, True, 'Correct output type')
            if key_output.shape == test_output.shape:
                autograder.score_test(test_id_shape, True, 'Correct output shape')

                score = torch.norm(key_output - test_output)
                autograder.score_test(test_id_val, score)
            else:
                autograder.score_test(test_id_shape, False, 'Incorrect output shape')
                autograder.score_test(test_id_val, False, 'Incorrect output')

            autograder.score_test(test_id_time, time_elapsed)
        else:
            autograder.score_test(test_id_type, False, 'Incorrect output type')
            autograder.score_test(test_id_shape, False, 'Incorrect output type')
            autograder.score_test(test_id_val, False, 'Incorrect output type')
            autograder.score_test(test_id_time, False, 'Incorrect output type')
    except Exception as ex:
            autograder.score_test(test_id_type, False, 'Error: '+str(ex))
            autograder.score_test(test_id_shape, False, 'Error: '+str(ex))
            autograder.score_test(test_id_val, False, 'Error: '+str(ex))
            autograder.score_test(test_id_time, False, 'Error: '+str(ex))

    print('problem 2 complete')

    #plt.plot(test_output[0])
    #plt.plot(sine)
    #plt.legend(['deriv', 'fct'])
    #plt.show()

    #ND test
    z = np.tile(np.linspace(0, 2*np.pi, 200), (200, 200, 1))
    y = np.moveaxis(z.copy(), 2, 1)
    x = np.moveaxis(z.copy(), 2, 0)
    fct = (np.sin(x) + np.cos(y))**2 + z**2

    key_output = key.compute_derivative(2*np.pi/2000, fct)

    # define tests
    test_id_shape = autograder.create_test('Problem 2, ND Output Shape', 1)
    test_id_val = autograder.create_test('Problem 2, ND Correct Output Values', 4, 0.01, 10)
    test_id_time = autograder.create_test('Problem 2, ND Time', 2, 1, 10)

    try: 
        time_start = time.perf_counter()
        test_output = cut.compute_derivative(2*np.pi/2000, fct)
        time_elapsed = time.perf_counter() - time_start
        print(time_elapsed)
        if type(test_output) == torch.Tensor:
            if key_output.shape == test_output.shape:
                autograder.score_test(test_id_shape, True, 'Correct output shape')

                score = torch.norm(key_output - test_output)
                autograder.score_test(test_id_val, score)
                autograder.score_test(test_id_time, time_elapsed)
            else:
                autograder.score_test(test_id_shape, False, 'Incorrect output shape')
                autograder.score_test(test_id_val, False, 'Incorrect output shape')
                autograder.score_test(test_id_time, False, 'Incorrect output shape')
        else:
            autograder.score_test(test_id_shape, False, 'Incorrect output type')
            autograder.score_test(test_id_val, False, 'Incorrect output type')
            autograder.score_test(test_id_time, False, 'Incorrect output type')
    except Exception as ex:
        autograder.score_test(test_id_shape, False, 'Error: '+str(ex))
        autograder.score_test(test_id_val, False, 'Error: '+str(ex))
        autograder.score_test(test_id_time, False, 'Error: '+str(ex))

    print('problem 2 ND complete')

    #plt.subplot(4,1,1)
    #plt.imshow(test_output[0,:,:,0])
    #plt.colorbar()
    #plt.subplot(4,1,2)
    #plt.imshow(test_output[1,:,:,0])
    #plt.colorbar()
    #plt.subplot(4,1,3)
    #plt.imshow(test_output[2,0,:,:])
    #plt.colorbar()
    #plt.subplot(4,1,4)
    #plt.imshow(fct[:,:,0])
    #plt.colorbar()
    #plt.show()

def test_sim_systems(cut, key, autograder):
    #seed for repeatability
    np.random.seed(1)

    init_cond = np.random.rand(1000,1,1001)
    A = np.random.rand(1000,1000)
    b = np.random.rand(1000)

    key_output = key.sim_systems(A, b, init_cond)

    #define tests
    test_id_type = autograder.create_test('Problem 1, Output Type', 1)
    test_id_shape = autograder.create_test('Problem 1, Output Shape', 1)
    test_id_val = autograder.create_test('Problem 1, Correct Output Values (RMSE)', 4, 0.01, 10)
    test_id_time = autograder.create_test('Problem 1, Time (sec)', 2, 1, 10)

    try:
        time_start = time.perf_counter()
        test_output = cut.sim_systems(A, b, init_cond)
        time_elapsed = time.perf_counter() - time_start
        print(time_elapsed)
        if type(test_output) == np.ndarray:
            autograder.score_test(test_id_type, True, 'Correct output type')
            if key_output.shape == test_output.shape:
                autograder.score_test(test_id_shape, True, 'Correct output shape')

                score = np.linalg.norm(key_output - test_output)
                autograder.score_test(test_id_val, score)
            else:
                autograder.score_test(test_id_shape, False, 'Incorrect output shape')
                autograder.score_test(test_id_val, False, 'Incorrect output shape')

            autograder.score_test(test_id_time, time_elapsed)
        else:
            autograder.score_test(test_id_type, False, 'Incorrect output type')
            autograder.score_test(test_id_shape, False, 'Incorrect output type')
            autograder.score_test(test_id_val, False, 'Incorrect output type')
            autograder.score_test(test_id_time, False, 'Incorrect output type')
    except Exception as ex:
        autograder.score_test(test_id_type, False, 'Error: '+str(ex))
        autograder.score_test(test_id_shape, False, 'Error: '+str(ex))
        autograder.score_test(test_id_val, False, 'Error: '+str(ex))
        autograder.score_test(test_id_time, False, 'Error: '+str(ex))

    print('problem 1 complete')


if __name__=='__main__':
    class_under_test = hw0_solution.HW0Solution()
    key = hw0_key.HW0Solution()
    autograder = Autograder('Problem Set 0')

    test_sim_systems(class_under_test, key, autograder)
    test_compute_derivative(class_under_test, key, autograder)
    autograder.write('/autograder/results/results.json')
