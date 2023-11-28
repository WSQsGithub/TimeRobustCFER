# TODOs
# [ ] Formula update
# [ ] Evaluation
# [ ] Plot Figure


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
    "prob_right": 1,
    "state0": np.array([0.65,0.35])
}

env = GridWorldEnv(ENV_CONFIG)
env.reset()
print(">> Initializing ...")
print(">> e_state = ", env.state)
# initializing the task

TASK_CONFIG = {
    "op_out" : "G",
    "op_in" : ["F", "F"],
    "T" : 10,
    "tau" : [10,10],
    "delta": 2,
    "beta" :    0.999,
    "scale":    [1,-0.9],
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


# %%

print(">>> Start Training with naive Q-Learning")

LEARN_CONFIG = {
    "MAX_EPISODE": 10000,
    "CHECK_INTERVAL": 1000,
    "EPS": [0.99, 0.01, 400],
    "LR" : [0.01, 0.01, 300],
    "GAMMA": 0.999,
    "BUFFER_SIZE": 1000,
    "REPLAY_PER" : 4,
    "BATCH_SIZE" : 64,
    "ALG" : "naive"
}

train_log = agent.learn(LEARN_CONFIG)
# save checkpoint
agent.QTable.save('../data/DSDA/Naive_tau10d2_grid10_1126.pkl')
scipy.io.savemat('../data/DSDA/Naive_tau10d2_grid10_1126.mat', train_log)
# plot training curve


TEST_CONFIG = {
    "MAX_EPISODE" : 500
}

test_log = agent.test(TEST_CONFIG)

# %%

print(">>> Start Training with CFER Q-Learning")

LEARN_CONFIG = {
    "MAX_EPISODE": 10,
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
