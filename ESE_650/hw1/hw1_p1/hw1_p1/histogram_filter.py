import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """
    
    def __init__(self):
        self.a = []
#        self.b = []

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
    
        ### Your Algorithm goes Below.
        '''
        
        
        transition_matrix = np.full((belief.shape[0],belief.shape[1],5), 0.0)
        observation_matrix = np.full((belief.shape[0],belief.shape[1]), 0.0)
#        transition_matrix_backward = np.full((20,20,5), 0.0)
        
        for i in range(0, belief.shape[0]):
            for j in range(0, belief.shape[1]):
                
# =============================================================================
#                 end_dest = (i-action[1], j+action[0])
#                 
#                 if (end_dest[0] < 0 or end_dest[0] > 19) or (end_dest[1] < 0 or end_dest[1] > 19):
#                     transition_matrix_backward[i,j] = [0,0,0,0,1]
#                     
#                 else:
#                     if action[0]==1:
#                         transition_matrix_backward[i,j] = [0,.9,0,0,.1]
#                     elif action[0]==-1:
#                         transition_matrix_backward[i,j] = [0,0,0,.9,.1]
#                     elif action[1]==1:
#                         transition_matrix_backward[i,j] = [.9,0,0,0,.1]
#                     elif action[1]==-1:
#                         transition_matrix_backward[i,j] = [0,0,.9,0,.1]
# =============================================================================


                if action[0]==1:
                    if j>0:
                        transition_matrix[i,j] = [0,0,0,.9,.1]
                    else:
                        transition_matrix[i,j] = [0,0,0,0,1]
                elif action[0]==-1:
                    if j<belief.shape[1]-1:
                        transition_matrix[i,j] = [0,.9,0,0,.1]
                    else:
                        transition_matrix[i,j] = [0,0,0,0,1]
                elif action[1]==1:
                    if i<belief.shape[0]-1:
                        transition_matrix[i,j] = [0,0,.9,0,.1]
                    else:
                        transition_matrix[i,j] = [0,0,0,0,1]
                elif action[1]==-1:
                    if i>0:
                        transition_matrix[i,j] = [.9,0,0,0,.1]
                    else:
                        transition_matrix[i,j] = [0,0,0,0,1]
                        
                if cmap[i,j] == observation:
                    observation_matrix[i,j] = .9
                else:
                    observation_matrix[i,j] = .1
                    
        
        if not self.a :
            self.a.append(np.multiply(belief, observation_matrix))
#            self.b.append(np.full((20,20), 1.0))
            a_next = np.multiply(belief, observation_matrix)
        
        else:
            a_next = np.full((belief.shape[0],belief.shape[1]), 0.0)
            for i in range(0,belief.shape[0]):
                for j in range(0,belief.shape[1]):
                    
                    a_mult = []
                    
                    indices = [(i-1, j), (i, j+1), (i+1, j), (i, j-1), (i,j)]
                    
                    counter = 0
                    
                    for x, y in indices:
                        if x<0 or x>belief.shape[0]-1 or y<0 or y>belief.shape[1]-1:
                            a_mult.append(0)
                        else:
                            a = self.a[len(self.a)-1][x,y]
                            t = transition_matrix[i,j][counter]
                            a_mult.append(a*t)
                        
                        
                        counter+=1
                    
                    a_next[i,j] = observation_matrix[i,j] * np.sum(a_mult)
                    
                    
# =============================================================================
#             b_past = np.full((20,20), 0.0)
#             for i in range(0,20):
#                 for j in range(0,20):
#                     
#                     b_mult = []
#                     
#                     
#                     for t in transition_matrix_backward[i,j]:
#                     
#                         b_mult.append((self.b[0][x,y]) * t)
#                         
#                     
#                     indices = [(i-1, j), (i, j+1), (i+1, j), (i, j-1), (i,j)]
#                     
#                     b_mult_two = []
#                     
#                     for x, y in indices:
#                         if x<0 or x>19 or y<0 or y>19:
#                             b_mult_two.append(0)
#                         else:
#                             b_mult_two.append(observation_matrix[x,y])
#                     
#                     
#                     b_past[i,j] = np.sum(np.multiply(b_mult, b_mult_two))
# =============================================================================
                    
            self.a.append(a_next)
#            self.b.insert(0, b_past)
                        
        to_return = a_next/np.sum(a_next)
        
        return np.asarray(to_return)
        
        