# %%
from typing import Optional, Tuple
import numpy as np
import gym
from gym import spaces
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Arrow
from random import random
import copy


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config):
        super(GridWorldEnv, self).__init__()
        self.gridsize = config['gridsize']
        self.dgrid = 1/self.gridsize
        self.goals = config['goals']
        self.obstacles = config['obstacles'] 
        self.prob_right = config['prob_right']
        self.beta = config['beta'] 
        self.scale = config['scale']
        
        
        self.action = ['stay', 'N', 'S', 'W', 'E', 'NW', 'SW', 'NE', 'SE']
        self.motion_uncertainty = np.array([[0,0,0],
                                            [0,5,7],
                                            [0,6,8],
                                            [0,5,6],
                                            [0,7,8],
                                            [0,1,3],
                                            [0,2,3],
                                            [0,1,4],
                                            [0,2,4]])
        
        self.move_dir = np.array([[0,0],
                                  [0,1],
                                  [0,-1],
                                  [-1,0],
                                  [1,0],
                                  [-1,1],
                                  [-1,-1],
                                  [1,1],
                                  [1,-1]])
        self.color = ['#A8BCDA', '#9F5751', '#7FA362']        
        
        # each robot has a action space of 9
        self.action_space = spaces.Discrete(9)
        # Observation space consists of position (px, py) and orientation (theta) for each robot.
        self.observation_space = spaces.MultiDiscrete([self.gridsize, self.gridsize])
        self.state = np.array([0,0])
                
        
    
    def reset(self, state=None):
        if not state is None:
            self.state = state
        else:
        # Reset the state of the environment to an initial state for each robot
            self.state = np.array([0,0]) # All robots start at the origin, facing right
        return self.state
    
    
    def step(self, action):
        cur_state = self.state
        
        if random.random()>self.prob_right: # action uncertainty
            uncertainty = random.choice(self.motion_uncertainty[action,:])
        else: 
            uncertainty = action
            
        next_state = cur_state + self.dgrid*self.move_dir[uncertainty,:]
        
        # stay before collision
        if self.checkCollision(next_state):
            next_state = copy.deepcopy(cur_state)
            done = 1
            # print('collide!', cur_state, self.action[action], next_state)
        else:
            next_state = copy.deepcopy(next_state)
            done = 0
            
        reward = 0 # reward is calculated in the traing loop
        self.state = next_state
        info = {
            'action': action,
            'uncertainty': uncertainty
            }
        return self.state, reward, done, info
    
    def render(self, trajectory=None):
         # create a figure
        fig, ax = plt.subplots(figsize=(5,5))
        
        # plot grid
        for i in range(self.grid_size+1):
            ax.axhline(y=i*self.dgrid, color='black', linewidth=0.5)
            ax.axvline(x=i*self.dgrid, color='black', linewidth=0.5)
            
        # plot region
        for i in range(len(self.region)):
            left = self.region[i,0]
            bottom = self.region[i,1]
            width = self.region[i,2]
            height = self.region[i,3]
            rect = patches.Rectangle((left, bottom), width, height, facecolor=self.color[i])
            ax.add_patch(rect)
            
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # plot obstacles
        if self.obstacle is not None:
            for i in range(len(self.obstacle)):
                left = self.obstacle[i,0]
                bottom = self.obstacle[i,1]
                width = self.obstacle[i,2]
                height = self.obstacle[i,3]
                rect = patches.Rectangle((left, bottom), width, height, facecolor='gray')
                ax.add_patch(rect)
            
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
    
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if trajectory is None:
            plt.show(block=False)
            return
        else:
            start_point = trajectory[0]
            plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=4)
            plt.scatter(trajectory[:, 0], trajectory[:,1], marker='.', s=200, label='Start')
            plt.scatter(start_point[0], start_point[1], marker='*', color='red', s=200, label='Start')
            plt.show(block=False)
    
    
    
        
    def checkCollision(self,state)-> int:
        # print(state,state - self.obstacle[:,0:1])
        
        if (np.min(state)<=0 or np.max(state)>=1):
            return 1
        if (self.obstacle is not None):
            for obs in self.obstacle:
                obs_x, obs_y, obs_width, obs_height = obs
                obs_top = obs_y + obs_height
                obs_right = obs_x + obs_width
                x, y = state
                if obs_x <= x <= obs_right and obs_y <= y <= obs_top:
                    return 1
        return 0


    
class MultiUnicycleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=10, height=10, num_robots=3):
        super(MultiUnicycleEnv, self).__init__()

        self.num_robots = num_robots  # Number of robots
        self.width = width
        self.height = height

        # Each robot has a speed 'v' and an angular velocity 'omega'.
        self.action_space = spaces.Box(
            low=np.tile(np.array([0, -np.pi / 2]), (self.num_robots, 1)),
            high=np.tile(np.array([0.75, np.pi / 2]), (self.num_robots, 1)),
            dtype=np.float32
        )

        # Observation space consists of position (px, py) and orientation (theta) for each robot.
        self.observation_space = spaces.Box(
            low=np.tile(np.array([0, 0, -np.pi]), (self.num_robots, 1)),
            high=np.tile(np.array([width, height, np.pi]), (self.num_robots, 1)),
            dtype=np.float32
        )

        self.state = None
        self.time_step = 0.2  # Time step for simulation

    def reset(self, state=None):
        if not state is None:
            self.state = state
        else:
        # Reset the state of the environment to an initial state for each robot
            self.state = np.zeros((self.num_robots, 3), dtype=np.float32)  # All robots start at the origin, facing right
        return self.state

    def step(self, actions):
        # Update the state of each robot based on its action
        new_states = []
        reward = np.zeros(self.num_robots)  # A vector of rewards, one per robot
        done = np.array([False] * self.num_robots)  # A vector of done flags, one per robot
        info = {}  # Additional info if needed
        
        for idx, (v, omega) in enumerate(actions):
            px, py, theta = self.state[idx]

            # dx/dt = v * cos(θ)
            # dy/dt = v * sin(θ)
            # dθ/dt = ω
            # ds/dt = [cos(θ) sin(θ) 1]^T [v, v, ω]
            # but there's nothing to learn in unicyle!

            # Update the robot's state using the unicycle model dynamics
            new_px = px + v * np.cos(theta) * self.time_step
            new_py = py + v * np.sin(theta) * self.time_step
            new_theta = theta + omega * self.time_step

            # Append the new state to the list of new states
            new_state = [new_px, new_py, new_theta]
            if self.check_collision(new_state):
                new_state = [px, py, new_theta]
            new_states.append(new_state)

            

            


        # Update the environment state with new states of all robots
        self.state = np.array(new_states)
        return self.state, reward, done, info
    
    def check_collision(self, state):
        print(state)
        if state[0] <= 0:
            return True
        if state[1] <= 0:
            return True
        if state[0] >= self.width:
            return True
        if state[1] >= self.height:
            return True

    def render(self, mode='human', close=False):
        if self.state is None:
            return None

        # Check if a figure is already open
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots()
            # Set the plot limits, assuming a square environment
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            self.ax.set_aspect('equal')
            # This keeps the arrow size consistent
            self.ax.autoscale(enable=False)
        
        for (px, py, theta) in self.state:
            # Calculate the end point of the arrow
            endx = px + 0.5 * np.cos(theta)
            endy = py + 0.5 * np.sin(theta)
            # Plot the arrow for each robot
            self.ax.add_patch(Arrow(px, py, endx - px, endy - py, width=0.1, color='r'))
        
        plt.pause(0.1)  # Pause for a brief moment to update the plot
        plt.draw()

        if close == True:
            plt.close()

# %%
# %matplotlib inline

if __name__ == '__main__':

    # Initialize the multi-robot environment
    env = MultiUnicycleEnv(width=10, height=10, num_robots=2)
    
    # Reset the environment and get the initial state
    state = env.reset(np.array([[5,5,0],[4,4,0]]))

    # Run a simple simulation loop
    for _ in range(100):
        # Generate some random actions for each robot


        actions = env.action_space.sample()
        # actions = np.array([[0.75,0.2]])
        state = env.state

        # Step the environment with the chosen actions
        next_state, reward, done, info = env.step(actions)

        env.render()

