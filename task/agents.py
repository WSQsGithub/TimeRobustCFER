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

    def add(self, exps):
        """ 添加一个或多个新的经验到缓冲区 """
        if isinstance(exps, list):
            self.memory.extend(exps)  # 如果是列表，添加所有经验
        else:
            self.memory.append(exps)  # 如果是单个经验，直接添加

    def sample(self, batch_size):
        """ 随机抽样一批经验 """
        return random.choices(self.memory, k=batch_size)

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
    
    def resetState(self):
        self.env.reset()
        self.task.reset()
        return self.getState()
    
    def test(self, config):
        n_trials = config["MAX_EPISODE"]
        episode_reward_log = np.zeros((1, n_trials))
        reward_log = np.zeros((self.task.H, n_trials ))
        e_state_log = np.zeros((self.task.H, len(self.env.state), n_trials))
        f_state_log = np.zeros((self.task.H , self.task.n_flag+1, n_trials,))
        rb_log = np.zeros((1,n_trials))
        
        # simulation for testing
        for episode in tqdm(range(n_trials), desc='# episode'):

            self.resetState()   

            # get current state
            episode_reward = 0
            for step in range(self.task.H):
                state = self.getState()    
                action = self.chooseAction(state, 0)
                
                new_e_state, _, done, info = self.env.step(action)
                new_f_state = self.task.updateFlag(self.task.sat4Pred(new_e_state), self.task.flag)
                
                reward = self.task.getReward(new_f_state[-1])
                episode_reward += reward
                
                next_state = self.getState() 
        
                
                reward_log[step,episode] = reward
                e_state_log[step,:,episode] = new_e_state
                f_state_log[step,:,episode] = new_f_state
                
                
            rb_log[0,episode] = self.task.getRobustness(e_state_log[:, :, episode])
            episode_reward_log[0,episode] = episode_reward
                
        log = dict()
        log['episode_reward'] = episode_reward_log
        log['reward'] = reward_log
        log['state']  = e_state_log
        log['f_state'] = f_state_log
        log['rb'] = rb_log
        log['p_sat'] = sum(rb_log[0]>0)/n_trials
        
        print(f"Satisfaction rate = {log['p_sat']}")
        
        return log    
        
    def evaluatePolicy(self, test_log):
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
        
        algorithm = config["ALG"]

        # naive q-learning without experience replay
        if algorithm == 'naive':
            train_log = self.learn_naive()
        elif algorithm == "CFER":
            train_log = self.learn_CFER()
        return train_log
    
    
    def decay_meta(self):
        '''
        We decay the exploration rate and learning rate expontially
        '''
        self.eps = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1*self.n_episodes_done/self.EPS_DECAY)
        self.lr = self.LR_END + (self.LR_START - self.LR_END) * math.exp(-1*self.n_episodes_done/self.LR_DECAY)

    def learn_naive(self):
        '''
        Naive Q-Learning
        '''
        try:
            # initialize a tensorboard directory
            writer = SummaryWriter(f"../tensorboard/{self.name}")
            # initialize a data collector
            
            train_log = dict()
            
            episode_reward_log = np.zeros((1,self.MAX_EPISODE))
            rb_log = np.zeros((1,self.MAX_EPISODE))

            p_bar = tqdm(range(self.MAX_EPISODE), desc="# episode")
            for episode in p_bar:
                # parameter decay

                self.decay_meta()
                p_bar.set_postfix_str(f"lr = {self.lr}, eps = {self.eps}")
                self.resetState()
                episode_reward = 0
                state_log = np.zeros((self.task.H, len(self.env.state)))
                
                # one learning iteraction
                for step in range(self.task.H):
                    state = self.getState()
                    print('>>> QTable Query: ', [self.QTable.get_q_value(state, a) for a in range(self.env.action_space.n)])  
                    action = self.chooseAction(state, self.eps)
                    print('>>> Chose action: ', action)
                    
                    new_e_state, _, done, info = self.env.step(action)
                    new_f_state = self.task.updateFlag(self.task.sat4Pred(new_e_state), self.task.flag)
                    print('>>> Get to new state:' , new_e_state, new_f_state)
                    
                    reward = self.task.getReward(new_f_state[-1])
                    print('Get reward: ', reward)
                    episode_reward += reward
                    
                    next_state = self.getState() 
                    
                    experience = Experience(state, action, reward, next_state, done)
                
                    self.QTable.update(experience, lr=self.lr, gamma=self.gamma)
                    print(">>> Update Q-table with Transition: ", experience)
                    print('>>> The new Q is:', self.QTable.get_q_value(state, action))
                
                episode_reward_log[0,episode] = episode_reward
                rb_log[0,episode] = self.task.getRobustness(state_log)
                self.n_episodes_done += 1
                
                # show curve every check interval
                if (episode%self.CHECK_INTERVAL==0) and episode:
                    print(">>> Episode: ", episode, "reward: ", np.mean(episode_reward_log[0][episode-self.CHECK_INTERVAL: episode]))
                    writer.add_scalar('naive', np.mean(episode_reward_log[0][episode-self.CHECK_INTERVAL: episode]))

        except KeyboardInterrupt:
            print('>>> Interupted at episode = ', episode)
            
        train_log['episode'] = episode+1
        train_log['episode_reward'] = episode_reward_log
        train_log['rb'] = rb_log
        train_log['sat'] = (rb_log>0)
        
        return train_log
    
    def learn_CFER(self):
        '''
        Learn with CFER
        '''
        try:
            # initialize a tensorboard directory
            writer = SummaryWriter(f"../tensorboard/{self.name}")
            # initialize a data collector
            
            train_log = dict()
            
            episode_reward_log = np.zeros((1,self.MAX_EPISODE))
            rb_log = np.zeros((1,self.MAX_EPISODE))

            p_bar = tqdm(range(self.MAX_EPISODE), desc="# episode")
            for episode in p_bar:
                # parameter decay

                self.decay_meta()
                p_bar.set_postfix_str(f"lr = {self.lr}, eps = {self.eps}")
                self.resetState()
                episode_reward = 0
                state_log = np.zeros((self.task.H, len(self.env.state)))
                
                # one learning iteraction
                for step in range(self.task.H):
                    state = self.getState()    
                    action = self.chooseAction(state, self.eps)
                    
                    new_e_state, _, done, info = self.env.step(action)
                    new_f_state = self.task.updateFlag(self.task.sat4Pred(new_e_state), self.task.flag)
                    
                    reward = self.task.getReward(new_f_state[-1])

                    episode_reward += reward
                    
                    next_state = self.getState() 
                    
                    experience = Experience(state, action, reward, next_state, done)
                    CFERs = self.task.generateCF(experience)
                    self.buffer.add(CFERs)
                    
                    if ~step%self.replay_period and step and len(self.buffer) > self.batch_size: # policy update
                        experiences = self.buffer.sample(batch_size=self.batch_size)
                        for experience in experiences:
                            self.QTable.update(experience, lr=self.lr, gamma=self.gamma)

                
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
        
        algorithm = config["ALG"]
        self.optimizer = optim.Adam(self.QNet.parameters(), lr=self.LR_START)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

        # naive q-learning without experience replay
        if algorithm == 'naive':
            train_log = self.learn_naive()
        elif algorithm == "CFER":
            train_log = self.learn_CFER()
        
        return train_log
    
    def learn_naive(self):
        '''
        Learn with CFER
        '''
        try:
            # initialize a tensorboard directory
            writer = SummaryWriter(f"../tensorboard/{self.name}")
            # initialize a data collector
            
            train_log = dict()
            
            episode_reward_log = np.zeros((1,self.MAX_EPISODE))
            rb_log = np.zeros((1,self.MAX_EPISODE))

            p_bar = tqdm(range(self.MAX_EPISODE), desc="# episode")
            for episode in p_bar:
                # parameter decay

                self.decay_meta()
                p_bar.set_postfix_str(f"lr = {self.lr}, eps = {self.eps}")
                self.resetState()
                episode_reward = 0
                state_log = np.zeros((self.task.H, len(self.env.state)))
                
                # one learning iteraction
                for step in range(self.task.H):
                    state = self.getState()    
                    action = self.chooseAction(state, self.eps)
                    
                    new_e_state, _, done, info = self.env.step(action)
                    new_f_state = self.task.updateFlag(self.task.sat4Pred(new_e_state), self.task.flag)
                    
                    reward = self.task.getReward(new_f_state)
                    episode_reward += reward
                    
                    next_state = self.getState() 
                    
                    experience = Experience(state, action, reward, next_state, done)
                    self.buffer.add(experience)
                    
                    if (step+1)%self.replay_period==0: # policy update
                        experiences = self.buffer.sample(batch_size=10)
                        experiences = self.buffer.sample(batch_size=10)
                        self.QNet.update(self.optimizer, experiences, gamma=0.999)
                        self.scheduler.step()
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
    
    def learn_CFER(self):
        '''
        Learn with CFER
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
                self.resetState()
                episode_reward = 0
                state_log = np.zeros((self.task.H, len(self.env.state)))
                
                # one learning iteraction
                for step in range(self.task.H):
                    state = self.getState()    
                    action = self.chooseAction(state, self.eps)
                    
                    new_e_state, _, done, info = self.env.step(action)
                    new_f_state = self.task.updateFlag(self.task.sat4Pred(new_e_state), self.task.flag)
                    
                    reward = self.task.getReward(new_f_state)
                    episode_reward += reward
                    
                    next_state = self.getState() 
                    
                    experience = Experience(state, action, reward, next_state, done)
                    CFERs = self.task.generateCF(experience)
                    self.buffer.add(CFERs)
                    
                    if (step+1)%self.replay_period==0: # policy update
                        experiences = self.buffer.sample(batch_size=10)
                        for experience in experiences:
                            self.QTable.update(experience, lr=self.lr, gamma=self.gamma)
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
        
class DDPGAgent(Agent):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.actor = Actor(len(self.getState()), self.env.action_space.shape[0])
        self.critic = Critic(len(self.getState()), self.env.action_space.shape[0])
        
        self.actor_target = Actor(len(self.getState()), self.env.action_space.shape[0])
        self.critic_target = Critic(len(self.getState()), self.env.action_space.shape[0])
        
        self.actor_target.load_state_dict(self.actor.state_dict()) # 软更新
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.gamma = config["GAMMA"]
        self.tau = config["TAU"]

    
    def chooseAction(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1))
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        return action
    
    def learn(self, config):
        pass
    
    def train(experiences):
        pass