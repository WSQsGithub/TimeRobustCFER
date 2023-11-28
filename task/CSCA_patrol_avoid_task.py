# %%
from utils import *
from agents import DDPGAgent, Experience
import gym
from formulas import Formula_task_GFG


# 获取所有可用的环境列表
# envs = gym.envs.registry

# # 打印环境名称
# for env_spec in envs:
#     print(env_spec.id)
    
print("Initializing ...")
env = gym.make('MountainCarContinuous-v0')
env.reset()
print(">> e_state = ", env.state)
# %%

TASK_CONFIG = {
    "op_out" : "G",
    "op_in" : ["F", "F","G"],
    "T" : 10,
    "tau" : [3,3,3],
    "goals" : None
}

task = Formula_task_GFG(TASK_CONFIG)
print(">> f_state = ", task.flag)

AGENT_CONFIG = {
    "name" : "patrolling_agent",
    "task"  : task,
    "env"   : env,
    "buffer_size" : 32,
    "GAMMA" : 0.99,
    "TAU": 0.005,
    "QTable_pth" : None
}


agent = DDPGAgent(AGENT_CONFIG)
print(">> state = ", agent.getState())

# %%
print(">>> Start Training")

actor_optimizer = optim.Adam(agent.actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(agent.actor.parameters(), lr=0.001)
actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=30, gamma=0.1)
critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=30, gamma=0.1)


for step in range(10):
    state = agent.getState()    
    action = agent.chooseAction(state)

    new_e_state, _, done, info, _ = env.step(action)
    new_f_state = task.updateFlag(new_e_state, task.flag)

    reward = task.getReward(new_f_state)

    next_state = agent.getState() 

    experience = Experience(state, action, reward, next_state, done)
    CFERs = task.generateCF(experience)
    agent.buffer.add(CFERs)


    if step+1%100==0:
        experiences = agent.buffer.sample(batch_size=64)

        state = torch.FloatTensor(experiences(state))
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_action = agent.actor_target(next_state)
            target_q = reward + (1 - done) * agent.gamma * agent.critic_target(next_state, next_action)
        current_q = agent.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update
        actor_loss = -agent.critic(state, agent.actor(state)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
            target_param.data.copy_(agent.TAU * param.data + (1 - agent.TAU) * target_param.data)
        for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
            target_param.data.copy_(agent.TAU * param.data + (1 - agent.TAU) * target_param.data)
# agent.QTable.update(experience, lr=0.01, gamma=0.999)
# print(">> Transition: ", experience)
    
