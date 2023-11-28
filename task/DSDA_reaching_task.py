# TODOs
# [x] Formula update
# [ ] Evaluation
# [x] Plot Figure


# %%
from utils import *
from agents import QLearningAgent, Experience
from environments import GridWorldEnv
from formulas import Formula_task_FG
# initializing the environment
ENV_CONFIG = {
    "gridsize": 5,
    "goals" : np.array([[0.6,0,0.4,0.4],[0,0.6,0.4,0.4],[0.6,0.6,0.4,0.4]]),
    # "obstacles":  np.array([[0,0.2,0.2,0.2],[0,0.4,0.4,0.2],[0.4,0,0.2,0.2],[0.6,0.4,0.4,0.2],[0.4,0.8,0.2,0.2]]),
    "obstacles": None,
    "prob_right": 1,
    "state0": np.array([0.3,0.1])
}

env = GridWorldEnv(ENV_CONFIG)
env.reset()
print(">> Initializing ...")
print(">> e_state = ", env.state)
# initializing the task

TASK_CONFIG = {
    "op_out" : "F",
    "op_in" : ["G"],
    "T" : 10,
    "tau" : [2],
    "delta": 0,
    "beta" :    0.999,
    "scale":    [1,-0.5],
    "goals" : ENV_CONFIG["goals"][2,:]
}


task = Formula_task_FG(TASK_CONFIG)
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

# %% 
# test environment
# # % test the environment
# ['stay', 'N', 'S', 'W', 'E', 'NW', 'SW', 'NE', 'SE']
agent.resetState()

print('-------------------------------------------------------------------------')
agent.task.updateFlag(task.sat4Pred([0.35, 0.15]), agent.task.flag)
print(">> state = ", agent.getState())

action = 1
state, _, done, info = agent.env.step(action)
agent.task.updateFlag(task.sat4Pred(state), agent.task.flag)
print(">> state = ", agent.getState(), ", reward = ", agent.task.getReward(agent.task.g))

action = 7
state, _, done, info = agent.env.step(action)
agent.task.updateFlag(task.sat4Pred(state), agent.task.flag)
print(">> state = ", agent.getState(), ", reward = ", agent.task.getReward(agent.task.g))

action = 1
state, _, done, info = agent.env.step(action)
agent.task.updateFlag(task.sat4Pred(state), agent.task.flag)
print(">> state = ", agent.getState(), ", reward = ", agent.task.getReward(agent.task.g))

action = 7
state, _, done, info = agent.env.step(action)
agent.task.updateFlag(task.sat4Pred(state), agent.task.flag)
print(">> state = ", agent.getState(), ", reward = ", agent.task.getReward(agent.task.g))

action = 7
state, _, done, info = agent.env.step(action)
agent.task.updateFlag(task.sat4Pred(state), agent.task.flag)
print(">> state = ", agent.getState(), ", reward = ", agent.task.getReward(agent.task.g))
action = 7
state, _, done, info = agent.env.step(action)
agent.task.updateFlag(task.sat4Pred(state), agent.task.flag)
print(">> state = ", agent.getState(), ", reward = ", agent.task.getReward(agent.task.g))

action = 7
state, _, done, info = agent.env.step(action)
agent.task.updateFlag(task.sat4Pred(state), agent.task.flag)
print(">> state = ", agent.getState(), ", reward = ", agent.task.getReward(agent.task.g))

action = 0
state, _, done, info = agent.env.step(action)
agent.task.updateFlag(task.sat4Pred(state), agent.task.flag)
print(">> state = ", agent.getState(), ", reward = ", agent.task.getReward(agent.task.g))

action = 0
state, _, done, info = agent.env.step(action)
agent.task.updateFlag(task.sat4Pred(state), agent.task.flag)
print(">> state = ", agent.getState(), ", reward = ", agent.task.getReward(agent.task.g))

# %%

print(">>> Start Training with naive Q-Learning")

LEARN_CONFIG = {
    "MAX_EPISODE": 100,
    "CHECK_INTERVAL": 500,
    "EPS": [0.99, 0.01, 400],
    "LR" : [0.01, 0.01, 300],
    "GAMMA": 0.999,
    "BUFFER_SIZE": 1000,
    "REPLAY_PER" : 4,
    "BATCH_SIZE" : 64,
    "ALG" : "naive"
}

train_log = agent.learn(LEARN_CONFIG)
# %%
reward_curve = train_log['episode_reward'][0]

smoothed_arr, lower_bounds, upper_bounds = curveData(reward_curve, 10)

# plot smoothed data
x = np.array(range(len(smoothed_arr)))*10
plt.plot(x, smoothed_arr,linewidth=2, label='naive')

plt.fill_between(x, lower_bounds, upper_bounds, alpha=0.2)
plt.show()

# save checkpoint
agent.QTable.save('../data/DSDA/Naive_tau10d2_grid10_1126.pkl')
scipy.io.savemat('../data/DSDA/Naive_tau10d2_grid10_1126.mat', train_log)
# plot training curve
# %%

TEST_CONFIG = {
    "MAX_EPISODE" : 500
}

test_log = agent.test(TEST_CONFIG)
best_trace_id = np.argmax(test_log['episode_reward'])
best_trace = test_log['state'][:,:,best_trace_id]

env.render(best_trace)

# %%

print(">>> Start Training with CFER Q-Learning")

LEARN_CONFIG = {
    "MAX_EPISODE": 400,
    "CHECK_INTERVAL": 1000,
    "EPS": [0.99, 0.01, 400],
    "LR" : [0.01, 0.01, 300],
    "GAMMA": 0.999,
    "BUFFER_SIZE": 10000,
    "REPLAY_PER" : 4,
    "BATCH_SIZE" : 256,
    "ALG" : "CFER"
}


train_log = agent.learn(LEARN_CONFIG)

reward_curve = train_log['episode_reward'][0]

smoothed_arr, lower_bounds, upper_bounds = curveData(reward_curve, 50)

# plot smoothed data
x = np.array(range(len(smoothed_arr)))*50
plt.plot(x, smoothed_arr,linewidth=2, label='naive')

plt.fill_between(x, lower_bounds, upper_bounds, alpha=0.2)
plt.show()


# save checkpoint
agent.QTable.save('../data/DSDA/CFER_tau10d2_grid10_1126.pkl')
scipy.io.savemat('../data/DSDA/CFER_tau10d2_grid10_1126.mat', train_log)
# plot training curve



TEST_CONFIG = {
    "MAX_EPISODE" : 500
}

test_log = agent.test(TEST_CONFIG)

best_trace_id = np.argmax(test_log['episode_reward'])
best_trace = test_log['state'][:,:,best_trace_id]

env.render(best_trace)
# %%
