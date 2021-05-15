import torch
import torch.nn.functional as F
import time
import random
import numpy as np
from collections import deque
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os

num_episodes = 1
num_game_each_episode = 1000
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.99
scores = [[],[]]
best_avg_score = 0
scores_average_window = 5
striker_solved_score = 1
defender_solved_score = -1

two_striker = False

# configure model
action_size_agent = 6
state_size = 14*8*3

def transfer_6_action(n):
    list_actions = [(1,0,0), (2,0,0), (0,1,0), (0,2,0), (0,0,1), (0,0,2)]
    return list_actions[n]

channel = EngineConfigurationChannel()
env = UnityEnvironment(file_name = './env/football', side_channels=[channel])
channel.set_configuration_parameters(time_scale = 2)
env.reset()

behavior_name_pink = list(env.behavior_specs)[0]
behavior_name_blue = list(env.behavior_specs)[1]



# Some Functions
def sensor_front_sig(data):
    player = []
    sensor_data = []
    for sensor in range(33):
        player.append(data[8 * sensor:(8 * sensor) + 8])

    for stack in range(3):
        sensor_data.append(player[11 * stack:(11 * stack) + 11])

    return sensor_data

def sensor_back_sig(data):
    player = []
    sensor_data = []
    for sensor in range(9):
        player.append(data[8 * sensor:(8 * sensor) + 8])

    for stack in range(3):
        sensor_data.append(player[3 * stack:(3 * stack) + 3])
    return sensor_data

from model import QNetwork
from replay_memory import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, behavior_name, index_player, replay_memory_size=1e4, batch_size=100, gamma=0.99,
                 learning_rate = 1e-4, target_tau=1e-3, update_rate=10, seed=0):
        self.state_size = state_size
        self.current_state = []
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)
        self.behavior_name = behavior_name
        self.index_player = index_player
        self.close_ball_reward = 0
        self.touch_ball_reward = 0

        """
        Now we define two models: 
        (a) one netwoek will be updated every (step % update_rate == 0),
        (b) A target network, with weights updated to equal to equal to the network (a) at a slower (target_tau) rate.
        """

        self.network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network =  QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr= self.learn_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step ( for updating every UPDATE_EVERY steps)
        self.t_step = 0
    def load_model(self, path_model, path_target = None):
        params = torch.load(path_model)
        #self.network.set_params(params)
        self.network.load_state_dict(torch.load(path_model))
        if path_target != None:
            self.target_network.load_state_dict(torch.load(path_target))

    def choose_action(self, state, eps = 0.0):
        eps = -1
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # return a number from 0 to action_size
        else:
            return random.choice(np.arange(self.action_size)) # return a number from 0 to action_size


    def Read(self):
        decision_steps, terminal_steps = env.get_steps(self.behavior_name)
        try:
            signal_front = np.array(sensor_front_sig(decision_steps.obs[0][self.index_player, :]))  # 3 x 11 x 8
            signal_back = np.array(sensor_back_sig(decision_steps.obs[1][self.index_player, :]))  # 3 x 3 x 8
            #pre_state = []
            signal_front =np.array(signal_front)
            #print(signal_front.shape)
            #print(signal_back.shape)
            r = np.concatenate((signal_front,signal_back),axis=1)
            #print(r.shape)
            #input('ff')
            #pre_state.extend(list(np.array(signal_front).flatten()))
            #pre_state.extend(list(np.array(signal_back).flatten()))
            #state = np.array(pre_state)
            self.current_state = r
            count_close_to_ball = 0
            count_touch_ball = 0
            count_back_touch = 0
            count_back_close = 0
            self.rew_d_to_our_post =0
            self.rew_for_ball_dist = -0.1
            # Front Observation
            for i in range(len(signal_front[0])):
                if signal_front[0][i][0] == 1.0:
                    count_close_to_ball+= 1
                    self.rew_for_ball_dist = max(0.3*(1 - signal_front[0][i][7]),self.rew_for_ball_dist)

                    # Kicked the ball at the front
                    if signal_front[0][i][7] <= 0.03:
                        count_touch_ball += 1

                if signal_front[0][i][1] == 1.0:
                    self.rew_d_to_our_post =-0.1
                if signal_front[0][i][2] == 1.0:
                    self.rew_d_to_our_post =0.1

            # Back observation
            for i in range(len(signal_back[0])):
                if signal_back[0][i][0] == 1.0:
                    count_back_close+= 1

                    # Touches the ball at the back 
                    if signal_back[0][i][7] <= 0.03:
                        count_back_touch += 0

            self.back_touch = 1 if count_back_touch > 0 else 0
            self.back_close = 1 if count_back_close > 0 else 0

            # add reward if kick the ball
            self.touch_ball_reward = 3 if count_touch_ball > 0 else 0
            # Penalize for back touching the ball
            if count_back_touch>0:
                self.touch_ball_reward= -1.5
            
            # Penalize if the ball is not in view
            self.close_ball_reward = 0.1 if count_close_to_ball > 0 else -0.1
            # Penalize if the ball is behind the agent
            if count_back_close >0:
                self.close_ball_reward = -0.1

            return self.current_state
        except:
            self.touch_ball_reward = 0
            self.close_ball_reward = 0
            
        return self.current_state



def process_reward_list_agressive_striker(reward_list):
    if reward_list[0] > 0:
        return 100.0
    elif reward_list[0] == -1.0 or reward_list[0] == -2.0: # get scored on
        return -100
    else:  # apply existing penalty
        return -0.01

def process_reward_list_defender(reward_list):
    if reward_list[0] == -1.0 or reward_list[0] == -2.0:
        return -100.0
    elif reward_list[0] > 0:
        return 100
    else: # apply existing reward
        return 0.2


agent_b1 = Agent(state_size = state_size, action_size = action_size_agent, behavior_name= behavior_name_blue, index_player= 0)
agent_b1.load_model("./result/b1_act_lastest","./result/b1_target_lastest")
agent_b2 = Agent(state_size = state_size, action_size = action_size_agent, behavior_name = behavior_name_blue, index_player= 1)
agent_b2.load_model("./result/b2_act_lastest","./result/b2_target_lastest")
agent_p1 = Agent(state_size = state_size, action_size = action_size_agent, behavior_name = behavior_name_pink, index_player= 0)
agent_p1.load_model("./result/p1_act_lastest","./result/p1_target_lastest")
agent_p2 = Agent(state_size = state_size, action_size = action_size_agent, behavior_name = behavior_name_pink, index_player= 1)
agent_p2.load_model("./result/p2_act_lastest","./result/p2_target_lastest")

np.set_printoptions(precision=3)

list_reward_p1 = []
list_reward_p2 = []
list_reward_b1 = []
list_reward_b2 = []

purple_score = 0
blue_score = 0

epsss=-1
for i_episode in range(1, num_episodes + 1):
    env.reset()
    state_p1 = agent_p1.Read()  
    state_p2 = agent_p2.Read()  
    
    state_b1 = agent_b1.Read()  
    state_b2 = agent_b2.Read()
    
    for i_game in range(num_game_each_episode):
        print(i_episode, i_game)
        epsss=-1

        action_b1 = agent_b1.choose_action(agent_b1.current_state,epsss) 
        array_action_b1 = transfer_6_action(action_b1)

        action_b2 = agent_b2.choose_action(agent_b2.current_state,epsss) 
        array_action_b2 = transfer_6_action(action_b2)
 
        action_p1 = agent_p1.choose_action(agent_p1.current_state,epsss) 
        array_action_p1 = transfer_6_action(action_p1)

        action_p2 = agent_p2.choose_action(agent_p2.current_state,epsss) 
        array_action_p2 = transfer_6_action(action_p2)

######################__Set actions here__##########################
        
        env.set_actions(behavior_name_blue, np.array([array_action_b1, array_action_b2]))
        env.set_actions(behavior_name_pink, np.array([array_action_p1, array_action_p2]))


        env.step()

        next_state_p1 = agent_p1.Read()
        next_state_p2 = agent_p2.Read()

        next_state_b1 = agent_b1.Read()
        next_state_b2 = agent_b2.Read()

        dec_p, terminal_p = env.get_steps(behavior_name_pink)
        dec_b, terminal_b = env.get_steps(behavior_name_blue)
        goal_reward_p1 = process_reward_list_agressive_striker(dec_p.reward)
        goal_reward_p2 = process_reward_list_defender(dec_p.reward)

        goal_reward_b1 = process_reward_list_agressive_striker(dec_b.reward)
        goal_reward_b2 = process_reward_list_defender(dec_b.reward)



        # Reward for Blue Player
        reward_b1 =  agent_b1.touch_ball_reward + goal_reward_b1 + agent_b1.close_ball_reward
        reward_b2 =  agent_b2.touch_ball_reward + goal_reward_b2 + agent_b2.close_ball_reward

        # Reward for Purple Player
        reward_p1 =  agent_p1.touch_ball_reward + goal_reward_p1 + agent_p1.close_ball_reward
        reward_p2 =  agent_p2.touch_ball_reward + goal_reward_p2 + agent_p2.close_ball_reward
        

        #print(reward_p1)
        #print(reward_p2)
        #input('rew')
        list_reward_p1.append(reward_p1)
        list_reward_p2.append(reward_p2)
        list_reward_b1.append(reward_b1)
        list_reward_b2.append(reward_b2)

        #agent_p1.model_step(state_p1, action_p1, reward_p1, next_state_p1)
        #agent_p2.model_step(state_p2, action_p2, reward_p2, next_state_p2)

        #agent_b1.model_step(state_b1, action_b1, reward_b1, next_state_b1)
        #agent_b2.model_step(state_b2, action_b2, reward_b2, next_state_b2)

        state_p1 = next_state_p1
        state_p2 = next_state_p2

        state_b1 = next_state_b1
        state_b2 = next_state_b2

        # check there is goal
        #====================================================Change me if you need to remove reseting env!!!====================================================#
        if dec_p.reward[0] > 0:
            print('GOALLLLLLLLLLLLLLLLLLLL')
            print('Purple Player Score')
            purple_score = purple_score + 1
            time.sleep(2)
            env.reset()

        if dec_b.reward[0] > 0:
            print('GOALLLLLLLLLLLLLLLLLLLL')
            print('Blue Player Score')
            blue_score = blue_score + 1
            time.sleep(2)
            env.reset()


    # Decrease epsilon for epsilon-greedy policy by decay rate
    epsilon = max(epsilon_min, epsilon_decay*epsilon)

    print(f'The final score is {blue_score} VS {purple_score}')
    if blue_score == purple_score:
        print('We have a Draw')
    elif blue_score > purple_score:
        print(f'Blue player won with {blue_score} against {purple_score} from Purple Player')
    elif purple_score > blue_score:
        print(f'Purple player won with {purple_score} against {blue_score} from Blue Player')
    else:
        print('Something is Wrong, Please contact the referee')
    
    print(f'The total reward for blue is {np.sum(list_reward_b1) + np.sum(list_reward_b2)} while the total reward for purple is {np.sum(list_reward_p1) + np.sum(list_reward_p2)}.')

env.close()









