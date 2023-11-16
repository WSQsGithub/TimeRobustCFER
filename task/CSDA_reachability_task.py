# %%
from utils import *
from formulas import Formula_task_FG
from agents import DQNAgent, Experience
import gym


print("Initializing ...")
env = gym.make("CartPole-v1")
env.reset()
print(">> e_state = ", env.state)

TASK_CONFIG = {
    "op_out" : "F",
    "op_in" : ["G"],
    "T" : 10,
    "tau" : [3,3],
    "goals" : np.array([0,0,1,1])
}

task = Formula_task_FG(TASK_CONFIG)
print(">> f_state = ", task.flag)

AGENT_CONFIG = {
    "name" : "mountaincar_dqn_agent",
    "task"  : task,
    "env" : env,
    "buffer_size": 32,
    "filename" : None
}

agent = DQNAgent(AGENT_CONFIG)
print(">> state = ", agent.getState())

print(">> Start training ... ")
optimizer = optim.Adam(agent.QNet.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for step in range(1000):
    state = agent.getState()
    action = agent.chooseAction(state,epsilon=0)
    
    new_e_state, _, done, info, _ = env.step(action)
    new_f_state = task.updateFlag(new_e_state, task.flag)
    
    reward = task.getReward(new_e_state)
    
    next_state = agent.getState()
    experience = Experience(state, action, reward, next_state, done)
    CFERs = task.generateCF(experience)
    agent.buffer.add(CFERs)

    if step+1%100==0:
        experiences = agent.buffer.sample(batch_size=10)
        agent.QNet.update(optimizer, experiences, gamma=0.999)
        scheduler.step()
    
    print(">> Qnet: ", agent.QNet.print_model_parameters())