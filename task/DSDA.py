# %% 
# Importing packages
from environments import GridWorldEnv
import numpy as np
from tqdm import tqdm

# Configurations
ENV_CONFIG = {
    "gridsize": 10,
    "goals" : np.array([[0.6,0,0.4,0.4],[0,0.6,0.4,0.4],[0.6,0.6,0.4,0.4]]),
    "obstacles":  np.array([[0,0.2,0.2,0.2],[0,0.4,0.4,0.2],[0.4,0,0.2,0.2],[0.6,0.4,0.4,0.2],[0.4,0.8,0.2,0.2]]),
    "prob_right": 0.91,
    "beta" :    0.999,
    "scale":    [1,0]}

env = GridWorldEnv(config = ENV_CONFIG)
# env.render()
# %%
# the training loop of a q-learning agent
# The task is: G[0,12)[F[0,3)(s ∈ A) ∧ F[0,3)(s ∈ B)],
# there A = goals[0,:], B = goals[1,:]

def learn():
    pass

# %%
class QAgent():
    def __init__(self, action_space) -> None:
        self.action_space = action_space