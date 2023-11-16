# %%
from utils import *
from agents import QLearningAgent, Experience
from environments import GridWorldEnv
from formulas import Formula_task_GF
# initializing the environment
ENV_CONFIG = {
    "gridsize": 10,
    "goals" : np.array([[0.6,0,0.4,0.4],[0,0.6,0.4,0.4],[0.6,0.6,0.4,0.4]]),
    "obstacles":  np.array([[0,0.2,0.2,0.2],[0,0.4,0.4,0.2],[0.4,0,0.2,0.2],[0.6,0.4,0.4,0.2],[0.4,0.8,0.2,0.2]]),
    "prob_right": 0.91,
    "beta" :    0.999,
    "scale":    [1,0]
}

env = GridWorldEnv(ENV_CONFIG)
env.reset([0.35,0.15])
print(">> Initializing ...")
print(">> e_state = ", env.state)
# initializing the task

TASK_CONFIG = {
    "op_out" : "G",
    "op_in" : ["F", "F"],
    "T" : 10,
    "tau" : [3,3],
    "goals" : ENV_CONFIG["goals"][0:2,:]
}


task = Formula_task_GF(TASK_CONFIG)
print(">> f_state = ", task.flag)

AGENT_CONFIG = {
    "name" : "patrolling_agent",
    "task"  : task,
    "env"   : env,
    "buffer_size" : 32,
    "QTable_pth" : None
}

agent = QLearningAgent(AGENT_CONFIG)
print(">> state = ", agent.getState())
for i in range(agent.env.action_space.n):
    agent.QTable.set_q_value(agent.getState(),i, 0)
agent.QTable.show_q_table(5)


print(">>> Start Training")

for step in range(10):
    state = agent.getState()    
    action = agent.chooseAction(state, 0.8)
    
    new_e_state, _, done, info = env.step(action)
    new_f_state = task.updateFlag(task.sat4Pred(new_e_state), task.flag)
    
    reward = task.getReward(new_f_state)
    
    next_state = agent.getState() 
    
    experience = Experience(state, action, reward, next_state, done)
    agent.QTable.update(experience, lr=0.01, gamma=0.999)
    print(">> Transition: ", experience)
    
agent.QTable.show_q_table(999)