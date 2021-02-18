import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']


    #### Test your code here
    
    initial_belief_state = np.full((20,20), 1/400)
    hf = HistogramFilter()
    
    belief = initial_belief_state
    mlps = []
    
    print('Iteration 0: Initial Belief State')
    print('Sum of belief probabilities: {}'.format(np.sum(belief)))
    print()
    
    for i in range(0,len(actions)):
        belief = hf.histogram_filter(cmap, belief, actions[i], observations[i])
        
        print('Iteration {}'.format(i+1))
        print('Sum of belief probabilities: {}'.format(np.sum(belief)))
        
        most_likely_position = [0,0]
        for x in range(0,20):
            for y in range(0, 20):
                if belief[x,y]>belief[most_likely_position[0], most_likely_position[1]]:
                    most_likely_position[0] = x
                    most_likely_position[1] = y
        mlps.append(most_likely_position)
        
        mlp_print = [most_likely_position[1], 19-most_likely_position[0]]
        print('Most Likely Position: {}'.format(mlp_print))
    
    
