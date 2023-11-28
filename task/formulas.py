from utils import *
from agents import Experience

class Formula: 
    def __init__(self, config) -> None:
        '''
        initialize instance with number of subformulas, the outer temporal operator and the subformula operators and horizons
        '''
        self.op_out = config["op_out"] # the outer temporal operator:G/F
        self.op_in = config["op_in"] # the inner temporal opertator for each subtask: [G,F]
        self.n_flag = len(self.op_in) # number of flag variable needed
        self.f = np.zeros(self.n_flag) # f_flag
        self.g = -1 # g_flag
        self.flag = np.concatenate([self.f, [self.g]]) # each flag variable
        
        
        self.T = config["T"] # time bound for outer temporal operator
        self.tau = config["tau"] # horizon for each subtask
        self.H = self.T + np.max(self.tau) # horizon of the overall formula
        
        self.delta = config["delta"]
        self.beta = config['beta'] 
        self.scale = config['scale']
        
        self.showFormula()
        
    def getState(self):
        self.flag = np.concatenate([self.f, [self.g]])
        return self.flag
        
    def showFormula(self) -> None:
        self.formula = self.op_out + f'[0,{self.T})('
        for i in range(self.n_flag):
            self.formula += self.op_in[i] + f'[0,{self.tau[i]}) phi_{i} '
        self.formula += ')'
        print('>> Phi = ', self.formula,', delta = ', self.delta)
    
    def reset(self):
        self.f = np.zeros(self.n_flag)
        self.g = -1
        self.flag = self.getState()
        
    def updateFlag(self, sat: list, flag: int) -> int:
        '''
        sat: a list that contains the true value of each subformula's predicate
        e.g. task.updateFlag(task.sat4Pred(state))
        '''
        # print('sat4Pred = ', sat)
        new_f = np.zeros(self.n_flag)
        for i in range(0, self.n_flag):
            if self.op_in[i] == 'G':
                if sat[i] == 1:
                    new_f[i] = min(flag[i]+1, self.tau[i])
                else:
                    new_f[i] = 0
            elif self.op_in[i] == 'F':
                if sat[i] == 0:
                    new_f[i] = max(flag[i]-1, 0)
                else:
                    new_f[i] = self.tau[i]

        if self.sat4All(new_f) == 1:
            new_g = min(self.g+1, self.delta)
        else:
            new_g = -1

        self.f = new_f
        self.g = new_g
        self.flag = self.getState()
        
        return self.flag
  
    def sat4Sub(self, flag) -> bool:
        '''
        return the satisfaction for each subformula as a list
        '''
        sat_sub = np.zeros(self.n_flag)
        for i in range(self.n_flag):
            if self.op_in[i] == 'G':
                sat_sub[i] = (flag[i]==self.tau[i])
            elif self.op_in[i] == 'F':
                sat_sub[i] = (flag[i]>0)
        return sat_sub
    
    def getTRB(self, g):
        return g==self.delta
    
    
    def getReward(self, new_g_state):
        # for F formula, return reward if 
        # sat = self.sat4All(self.sat4Sub(new_f_state))
        sat = self.getTRB(new_g_state)
        
        if self.op_out == "G":
            # return self.scale[0]*(-np.exp(-self.beta*sat))+ self.scale[1]
            return self.scale[0]*sat + self.scale[1]
        
        elif self.op_out == "F":
            
            # return self.scale[0]*(np.exp(self.beta*sat))+ self.scale[1]
            return self.scale[0]*sat + self.scale[1]
        
    
    def generateCF(self, experience):
        action = experience.action
        e_state = experience.state[:-self.n_flag-1]
        next_e_state = experience.next_state[:-self.n_flag-1]
        cf_states = list(itertools.product(*[range(t) for t in np.concatenate([self.tau,[self.delta]])]))
        
        experiences = []
        
        for cf_state in cf_states:
            cf_state = np.array(cf_state)
            
            new_cf_state = self.updateFlag(self.sat4Pred(next_e_state), cf_state) 

            reward = self.getReward(new_cf_state[-1]) 
            
            experiences.append(Experience(np.concatenate([e_state, cf_state]), action, reward, np.concatenate([next_e_state, new_cf_state]), 0))
        
        return experiences
        
class Formula_task_GFG(Formula):
    def __init__(self, config) -> None:
        super().__init__(config=config)
        
        
class Formula_task_FG(Formula):
    # phi = F[0,12](G[0,3] x in A)
    def __init__(self, config):
        super().__init__(config=config)
        target = config["goals"]
        self.A = np.array([target[0:2],target[0:2]+target[2:4]])
        
    def sat4Pred(self,state) -> list:
        '''
        define the satisfaction of each predicate in the subformulas
        return a list of n_flag variables
        '''
        sat  = np.zeros(self.n_flag)
        # x in A
        sat[0] = np.min(state>self.A[0,:]) and np.min(state<self.A[1,:])
        return sat
        
    def sat4All(self, flag) -> int:
        '''
        define the satisfaction for the whole inner formula
        '''
        return np.min(self.sat4Sub(flag))
    
    def viewFlag(self):
        print('flag = ', self.flag, 'sat4sub = ', self.sat4Sub(), 'sat4All = ', self.sat4All()) 
        
    def getRobustness(self, trace) -> float:
        # the overall formula is F[0,T)G[0,tau[0]) x in A
        # print(np.shape(trace))
        r_sub = np.zeros(self.T)
        for t in range(self.T):
            r_sub[t] = np.min(np.minimum(np.min(trace[t:t+self.tau[0], :] - self.A[0,:],1), np.min(self.A[1,:] - trace[t:t+self.tau[0], :],1)))
        return np.max(r_sub)
           
# class Formula_task_FG(Formula):
#     # phi = F[0,12](G[0,3] x in A)
#     def __init__(self, op_out, op_in, T, tau, target):
#         super().__init__(op_out, op_in, T, tau)
#         self.A = np.array([target[0:2],target[0:2]+target[2:4]])
        
#     def sat4Pred(self,state) -> list:
#         '''
#         define the satisfaction of each predicate in the subformulas
#         return a list of n_flag variables
#         '''
#         sat  = np.zeros(self.n_flag)
#         # x in A
#         sat[0] = np.min(state>self.A[0,:]) and np.min(state<self.A[1,:])
#         return sat
        
#     def sat4All(self, flag) -> int:
#         '''
#         define the satisfaction for the whole inner formula
#         '''
#         return np.min(self.sat4Sub(flag))
    
#     def viewFlag(self):
#         print('flag = ', self.flag, 'sat4sub = ', self.sat4Sub(), 'sat4All = ', self.sat4All()) 
        
#     def getRobustness(self, trace) -> float:
#         # the overall formula is F[0,T)G[0,tau[0]) x in A
#         # print(np.shape(trace))
#         r_sub = np.zeros(self.T)
#         for t in range(self.T):
#             r_sub[t] = np.min(np.minimum(np.min(trace[t:t+self.tau[0], :] - self.A[0,:],1), np.min(self.A[1,:] - trace[t:t+self.tau[0], :],1)))
#         return np.max(r_sub)
        
        
class Formula_task_FF(Formula):
    # phi = F[0,T)(F[0,tau[0]) x in A)
    def __init__(self, op_out, op_in, T, tau, target):
        super().__init__(op_out, op_in, T, tau)
        self.A = np.array([target[0:2],target[0:2]+target[2:4]])
        
    def sat4Pred(self, state) -> list:
        sat = np.zeros(self.n_flag)
        sat[0] = np.min(state>self.A[0,:]) and np.min(state<self.A[1,:])
        return sat
    
    def sat4All(self, flag) -> int:
        return np.min(self.sat4Sub(flag))
    
    def viewFlag(self):
        print('flag = ', self.flag, 'sat4sub = ', self.sat4Sub(), 'sat4All = ', self.sat4All()) 
        
    def getRobustness(self, trace) -> float:
        r_sub = np.zeros(self.T)
        for t in range(self.T):
            r_sub[t] = np.max(np.minimum(np.min(trace[t:t+self.tau[0], :] - self.A[0,:],1), np.min(self.A[1,:] - trace[t:t+self.tau[0], :],1)))
        return np.max(r_sub)

    
class Formula_task_GF(Formula):
    # phi = G[0,T)(F[0,tau[0]) x in A and F[0,tau[1]) x in B)
    def __init__(self, config):
        super().__init__(config)
        target = config['goals']
        self.A = np.array([target[0,0:2],target[0,0:2]+target[0,2:4]])
        self.B = np.array([target[1,0:2],target[1,0:2]+target[1,2:4]])
        
    def sat4Pred(self,state) -> list:
        '''
        define the satisfaction of each predicate in the subformulas
        return a list of n_flag variables
        '''
        sat  = np.zeros(self.n_flag)
        # x in A
        sat[0] = np.min(state>self.A[0,:]) and np.min(state<self.A[1,:])
        # x in B
        sat[1] = np.min(state>self.B[0,:]) and np.min(state<self.B[1,:])
        
        return sat
        
    def sat4All(self, flag) -> int:
        '''
        define the satisfaction for the whole inner formula
        '''

        return np.min(self.sat4Sub(flag))

    
    def viewFlag(self):
        print('flag = ', self.flag, 'sat4sub = ', self.sat4Sub(), 'sat4All = ', self.sat4All()) 

    def getRobustness(self, trace) -> float:
        # phi = G[0,T)(F[0,tau[0]) x in A and F[0,tau[1]) x in B)
        r_sub_A = np.zeros(self.T)
        r_sub_B  =np.zeros(self.T)
        r_sub = np.zeros(self.T)
        for t in range(self.T):
            # might be something wrong with it
            r_sub_A[t] = np.max(np.minimum(np.min(trace[t:t+self.tau[0], :] -self.A[0,:],1), np.min(self.A[1,:] - trace[t:t+self.tau[0], :],1)))
            r_sub_B[t] = np.max(np.minimum(np.min(trace[t:t+self.tau[1], :] -self.B[0,:],1), np.min(self.B[1,:] - trace[t:t+self.tau[1], :],1)))
            r_sub[t] = min(r_sub_A[t],r_sub_B[t])

            # _ = input("Press [enter] to continue.")
        return np.min(r_sub)