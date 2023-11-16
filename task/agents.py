# %% 
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils import *
from nets import *

# 定义一个简单的命名元组来存储经验
Experience = namedtuple("Experience", field_names=["state","action", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 缓冲区的最大容量
        self.memory = deque(maxlen=capacity)  # 使用deque作为内部存储，当达到最大容量时将自动弃置旧数据

    def add(self, exp):
        """ 添加一个新的经验到缓冲区 """
        self.memory.append(exp)

    def sample(self, batch_size):
        """ 随机抽样一批经验 """
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        """ 返回当前缓冲区的大小 """
        return len(self.memory)
    
    

class Agent():
    def __init__(self, config) -> None:
        self.name = config["name"]
        self.task = config["task"]
        self.env = config["env"]
        self.state = self.getState()
        self.buffer = ReplayBuffer(config["buffer_size"])        
        
        
    def getState(self):
        self.state  = np.concatenate([self.env.state, self.task.flag])
        return self.state
    
    def resetState(self, e_state):
        self.env.reset(e_state)
        self.task.state = task.s0
        return self.getState()
        
    def evaluatePolicy(self):
        pass



class QLearningAgent(Agent):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.QTable = QTable(actions=['stay', 'N', 'S', 'W', 'E', 'NW', 'SW', 'NE', 'SE'][:self.env.action_space.n], filename=config["QTable_pth"])
        
    def chooseAction(self,state, epsilon) -> int:
        '''choose action based on e-greedy method'''
        if random.random() < epsilon:
            return self.env.action_space.sample()
        q_values = [self.QTable.get_q_value(state, a) for a in range(self.env.action_space.n)]
        max_q_value = max(q_values)
        actions_with_max_q_value = [a for a, q in zip( range(self.env.action_space.n), q_values) if q == max_q_value]
        return random.choice(actions_with_max_q_value)
        
    def learn(self, config):
        
        # load in the learning configuration:
        self.MAX_EPISODE = config['MAX_EPISODE']
        self.CHECK_INTERVAL = config['CHECK_INTERVAL']

        # learning parameters
        [self.EPS_START, self.EPS_END, self.EPS_DECAY] = config['EPS']
        [self.LR_START, self.LR_END, self.LR_DECAY] = config['LR']
        self.gamma = config['GAMMA']

        print(f'>>> See Tensorboard at ../tensorboard/{self.name}')
        
        self.buffer = ReplayBuffer(config["BUFFER_SIZE"])
        self.replay_period = config['REPLAY_PER']
        self.batch_size = config['BATCH_SIZE']
        
        # initialize learning
        self.n_updates_done = 0
        self.n_episodes_done = 0
        
        self.update_Meta(self.EPS_START, self.LR_START)
        
        algorithm = config["ALG"]

        # naive q-learning without experience replay
        if algorithm == 'naive':
            train_log = self.learn_naive(config["env"], config["task"])
        
        return train_log
    
    
    def decay_meta(self):
        '''
        We decay the exploration rate and learning rate expontially
        '''
        self.eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1*self.n_episodes_done/self.EPS_DECAY)
        self.lr = self.LR_END + (self.LR_START - self.LR_END) * math.exp(-1*self.n_episodes_done/self.LR_DECAY)

    def learn_naive(self, env, task):
        '''
        Naive Q-Learning without experience replay
        '''
        try:
            # initialize a tensorboard directory
            writer = SummaryWriter(f"../tensorboard/{self.name}")
            # initialize a data collector
            
            train_log = dict()
            
            episode_reward_log = np.zeros((1,self.MAX_EPISODE))
            # reward_log = np.zeros((self.task.H ,self.MAX_EPISODE))
            # state_log = np.zeros((self.task.H ,2,self.MAX_EPISODE))
            # f_state_log = np.zeros((self.task.H , self.task.n_flag,self.MAX_EPISODE))
            rb_log = np.zeros((1,self.MAX_EPISODE))

            p_bar = tqdm(range(self.MAX_EPISODE), desc="# episode")
            for episode in p_bar:
                # parameter decay

                self.decay_meta()
                p_bar.set_postfix_str(f"lr = {self.lr}, eps = {self.eps}")
                self.resetState(env, task)
                episode_reward = 0
                state_log = np.zeros((task.H, len(env.state)))
                
                # one learning iteraction
                for step in range(self.task.H):
                    state = agent.getState(env, task)    
                    action = agent.chooseAction(state, self.eps)
                    
                    new_e_state, _, done, info = env.step(action)
                    new_f_state = task.updateFlag(task.sat4Pred(new_e_state), task.flag)
                    
                    reward = task.getReward(new_f_state)
                    episode_reward += reward
                    
                    next_state = agent.getState(env, task) 
                    
                    experience = Experience(state, action, reward, next_state, done)
                    agent.QTable.update(experience, lr=self.lr, gamma=self.gamma)
                    # print(">> Transition: ", experience)
                
                episode_reward_log[0,episode] = episode_reward
                rb_log[0,episode] = self.task.getRobustness(state_log)
                self.n_episodes_done += 1
                
                # show curve every check interval
                if ~episode%self.CHECK_INTERVAL and episode:
                    writer.add_scalar('naive', np.mean(episode_reward_log[0][episode-self.CHECK_INTERVAL: episode]))

        except KeyboardInterrupt:
            print('>>> Interupted at episode = ', episode)
            
        train_log['episode'] = episode+1
        train_log['episode_reward'] = episode_reward_log
        train_log['rb'] = rb_log
        train_log['sat'] = (rb_log>0)
        
        return train_log
    

class DQNAgent(Agent):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.QNet = QNetwork(state_size=len(self.getState()),action_size=self.env.action_space.n, hidden_sizes=[32])
        if config["filename"] is not None:
            self.QNet.load(config["filename"])
            
    def chooseAction(self,state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.QNet(state)
            return torch.argmax(q_values).item()


if __name__ == "__main__":
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
        "QTable_pth" : None
    }
    
    agent = QLearningAgent(AGENT_CONFIG)
    print(">> state = ", agent.getState())
    for i in range(agent.env.action_space.n):
        agent.QTable.set_q_value(agent.getState(),i, 0)
    agent.QTable.show_q_table(5)
    
    
    print(">>> Start Training")

    for step in range(100):
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