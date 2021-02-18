import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution
        self.Observation_matrix = [[.4, .1, .5], [.1, .5, .4]]
    
    def forward(self):

        alpha = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        
        alpha[0] = np.multiply(self.Initial_distribution, [m[self.Observations[0]] for m in self.Observation_matrix])
        
        for i in range(1, len(alpha)):
            prev_a = alpha[i-1]
            new_a = []
            
            for j in range(0,2):
                a_sum = 0
                m = self.Observation_matrix[j][self.Observations[i]]
                for k in range(0, len(prev_a)):
                    a_sum += prev_a[k] * self.Transition[k,j]
                new_a.append(m*a_sum)
            
            alpha[i] = new_a

        return alpha
    
    def backward(self):
    
        beta = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        
        beta[beta.shape[0]-1] = [1,1]
        
        for i in range(len(beta)-2, -1, -1):
            next_b = beta[i+1]
            new_b = []
            
            for j in range(0,2):
                b_sum = 0
                for k in range(0,2):
                    b_val = next_b[k]
                    t = self.Transition[j,k]
                    p = self.Observations[i+1]
                    m = self.Observation_matrix[k][p]
                    b_sum += (b_val*t*m)
                new_b.append(b_sum)
            
            beta[i] = new_b
                    

        return beta
    
    def gamma_comp(self, alpha, beta):
        
        gamma = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        
        alpha_sum = np.sum(alpha[len(alpha)-1])
        
        for i in range(0, 20):
            for j in range(0,2):
                gamma[i,j] = (alpha[i,j] * beta[i,j])/(alpha_sum)

        return gamma
    
    def xi_comp(self, alpha, beta, gamma):
        
        xi = np.zeros((self.Observations.shape[0]-1, self.Transition.shape[0], self.Transition.shape[0]))
        
        for i in range(0, xi.shape[0]):
            for j in range(0, xi.shape[1]):
                for k in range(0, xi.shape[2]):
                    a = alpha[i,j]
                    t = self.Transition[j,k]
                    m = self.Observation_matrix[k][self.Observations[i+1]]
                    b = beta[i+1, k]
                    
                    xi[i,j,k] = a*t*m*b
            xi[i] = xi[i]/np.sum(xi[i])    
              
        xi_compare = np.zeros((self.Observations.shape[0]-1, self.Transition.shape[0], self.Transition.shape[0]))
                      
        for i in range(0, xi_compare.shape[0]):
            for j in range(0, xi_compare.shape[1]):
                for k in range(0, xi_compare.shape[2]):
                    xi_compare[i,j,k] = gamma[i,j] * gamma[i+1,k]
        
        if not (np.allclose(xi[0], xi_compare[0])):
            print('xis dont match')
        
        return xi

    def update(self, alpha, beta, gamma, xi):    

        new_init_state = np.zeros_like(self.Initial_distribution)
        T_prime = np.zeros_like(self.Transition)
        M_prime = np.zeros_like(self.Emission)
        
        new_init_state = gamma[0]
        
        for i in range(0, T_prime.shape[0]):
            for j in range(0, T_prime.shape[1]):
                n = 0
                d = 0 
                for k in range(0, xi.shape[0]):
                    n += xi[k,i,j]
                    d += gamma[k, i]
                T_prime[i,j] = n/d
        
        for i in range (0, M_prime.shape[0]):
            for j in range(0, M_prime.shape[1]):
                n = 0
                d = 0
                for k in range(0, gamma.shape[0]):
                    if self.Observations[k] == j:
                        n += gamma[k,i]
                    d += gamma[k, i]
                M_prime[i,j] = n/d
        
        return T_prime, M_prime, new_init_state
    
    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = np.array([0.])
        P_prime = np.array([0.])
        
        deltas_orig = []
        parents_orig = []
        deltas_prime = []
        parents_prime = []
        
        for i in range(0, alpha.shape[0]):
            o1 = []
            o2 = []
            p1 = []
            p2 = []
            
            if (i==0) :
                for j in range(0, self.Initial_distribution.shape[0]):
                    o1.append((self.Initial_distribution[j] * self.Observation_matrix[j][self.Observations[i]]))
                    o2.append(None)
                    p1.append((new_init_state[j] * M_prime[j][self.Observations[i]]))
                    p2.append(None)
            
            else:
                for j in range(0, self.Initial_distribution.shape[0]):
                    max_delt_orig = 0
                    max_delt_prime = 0
                    argmax_orig = None
                    argmax_prime = None
                    argmax_orig_p = 0
                    argmax_prime_p = 0
                    for k in range(0, self.Transition.shape[1]):
                        delt_o_k_plus_1 = deltas_orig[i-1][k] * self.Transition[k][j] * self.Observation_matrix[j][self.Observations[i]]
                        par_o_k_plus_1 = deltas_orig[i-1][k] * self.Transition[k][j]
                        
                        delt_p_k_plus_1 = deltas_prime[i-1][k] * T_prime[k][j] * M_prime[j][self.Observations[i]]
                        par_p_k_plus_1 = deltas_prime[i-1][k] * T_prime[k][j]
                        
                        max_delt_orig = max(delt_o_k_plus_1, max_delt_orig)
                        if par_o_k_plus_1 > argmax_orig_p :
                            argmax_orig_p = par_o_k_plus_1
                            argmax_orig = k
                            
                        max_delt_prime = max(delt_p_k_plus_1, max_delt_prime)
                        if par_p_k_plus_1 > argmax_prime_p :
                            argmax_prime_p = par_p_k_plus_1
                            argmax_prime = k
                        
                    o1.append(max_delt_orig)
                    o2.append(argmax_orig)
                    p1.append(max_delt_prime)
                    p2.append(argmax_prime)
                    
            deltas_orig.append(o1)
            parents_orig.append(o2)
            deltas_prime.append(p1)
            parents_prime.append(p2)
                    
                
        orig_likely_final = None
        prime_likely_final = None
        orig_final_prob = 0
        prime_final_prob = 0
        
        for x in range(0, len(deltas_orig[0])):
            if deltas_orig[len(deltas_orig)-1][x] > orig_final_prob :
                orig_final_prob = deltas_orig[len(deltas_orig)-1][x]
                orig_likely_final = x
                
            if deltas_prime[len(deltas_prime)-1][x] > prime_final_prob :
                prime_final_prob = deltas_prime[len(deltas_prime)-1][x]
                prime_likely_final = x
        
        P_original[0] = deltas_orig[len(deltas_orig)-1][orig_likely_final]
        P_prime[0] = deltas_prime[len(deltas_prime)-1][prime_likely_final]
        

        return P_original, P_prime
    
# Problem (a)
V = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1])

# # Transition Probabilities
a = np.array(((0.5, 0.5), (0.5, 0.5)))
# # Emission Probabilities
b = np.array(((0.4, 0.1, 0.5), (0.1, 0.5, 0.4)))
# # Equal Probabilities for the initial distribution
initial_distribution = np.array((0.5, 0.5))
hmm = HMM(Observations=V, Transition=a, Emission=b, Initial_distribution=initial_distribution)

alpha = hmm.forward()
beta = hmm.backward()
gamma = hmm.gamma_comp(alpha, beta)
xi = hmm.xi_comp(alpha, beta, gamma)
t_prime, m_prime, new_init_state = hmm.update(alpha, beta, gamma, xi)
p_orig, p_prime = hmm.trajectory_probability(alpha, beta, t_prime, m_prime, new_init_state)

print('Alphas')
print()
print(alpha)
print()
print()
print('Betas')
print()
print(beta)
print()
print()
print('Gammas')
print()
print(gamma)
print()
print()

state_preds = []
for x, y in gamma:
    if x > y :
        state_preds.append('LA')
    else:
        state_preds.append('NY')
print('State Preds')
print()
print(state_preds)
print()
print()

print('Xis')
print()
print(xi)
print()
print()
print('T primes')
print()
print(t_prime)
print()
print()
print('M_prime')
print()
print(m_prime)
print()
print()
print('New Init State')
print()
print(new_init_state)
print()
print()
print('P orig')
print()
print(p_orig)
print()
print()
print('P prime')
print()
print(p_prime)


