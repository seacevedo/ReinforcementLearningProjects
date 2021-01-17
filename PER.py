import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import copy 
from gym.wrappers import FrameStack
from gym.wrappers import AtariPreprocessing
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class ConvNet(nn.Module):
  def __init__(self, learning_rate, num_actions, input_dims):
    super(ConvNet, self).__init__()

    self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(3136, 512)
    self.fc2 = nn.Linear(512, num_actions)

    self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
    self.loss = nn.MSELoss()
    self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, state):
    layer1 = F.relu(self.conv1(state))
    layer2 = F.relu(self.conv2(layer1))
    layer3 = F.relu(self.conv3(layer2))
    flat = self.flat(layer3)
    layer4 = F.relu(self.fc1(flat))
    actions = self.fc2(layer4)
    return actions

class ReplayBuffer:
    def __init__(self, maxlen):
        self.experiences = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
    def add(self, experience):
        self.experiences.append(experience)
        self.priorities.append(max(self.priorities, default=1))
    def sample(self, alpha, beta, num_batch):
        exp_priorities = np.power(self.priorities, alpha)
        probs = exp_priorities / sum(exp_priorities)
        indices = np.random.choice(len(self.experiences), size=num_batch, p=probs)
        weights = np.power(len(probs[indices]) * probs[indices], -beta)
        importance_weights = weights / max(weights)
        batch = [self.experiences[index] for index in indices]
        states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
        return states, actions, rewards, next_states, dones, indices, importance_weights
    def update_priorities(self, indices, errors, offset=0.01):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

class Agent:
  def __init__(self, learning_rate, alpha, num_actions, input_dims, gamma=0.95, epsilon=1.0, epsilon_min=0.01, maxlen=50000):
    self.gamma = gamma
    self.maxlen = maxlen
    self.learning_rate = learning_rate
    self.alpha = alpha
    self.num_actions = num_actions
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.input_dims = input_dims

    self.Q = ConvNet(learning_rate=self.learning_rate, num_actions=self.num_actions, input_dims=self.input_dims)
    self.Q_target = copy.deepcopy(self.Q)

    self.replay_buffer = ReplayBuffer(maxlen=self.maxlen)

  def choose_action(self, observation):
    if np.random.random() < self.epsilon:
      action = np.random.choice(self.num_actions)
    else:
      state = T.tensor(observation, dtype=T.float).to(self.Q.device) / 255
      state = state.unsqueeze(0)
      actions = self.Q.forward(state)
      action = T.argmax(actions).item()
    return action


  def learn(self, num_batch):
    if len(self.replay_buffer.experiences) < num_batch:
      return
    self.Q.optimizer.zero_grad()
    states, actions, rewards, next_states, dones, indices, importance_weights = self.replay_buffer.sample(alpha=self.alpha, beta=1-(self.epsilon/2), num_batch=num_batch)
    states = T.tensor(states, dtype=T.float).to(self.Q.device) / 255
    actions = T.LongTensor(actions).to(self.Q.device)
    rewards = T.tensor(rewards).to(self.Q.device)
    next_states = T.tensor(next_states, dtype=T.float).to(self.Q.device) / 255
    dones = T.LongTensor(dones).to(self.Q.device)
    importance_weights = T.tensor(importance_weights, dtype=T.float).to(self.Q.device)

    q_pred = self.Q.forward(states)[range(states.shape[0]), actions]
    best_next_actions = self.Q.forward(next_states).argmax(dim=1)
    q_next = self.Q_target.forward(next_states)[range(states.shape[0]), best_next_actions]

    q_target = (rewards + (1 - dones) * self.gamma * q_next)
    errors = q_target - q_pred 

    loss = (errors).pow(2) * importance_weights
    loss = loss.mean()
    loss.backward()
    self.replay_buffer.update_priorities(indices, errors.data.cpu().numpy())
    self.Q.optimizer.step()


env = gym.make('AsterixNoFrameskip-v4')
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
agent = Agent(learning_rate=0.0001, alpha=0.7, num_actions=env.action_space.n, input_dims=env.observation_space.shape)
num_batch = 32
frames = 0
rand_frames = 0
scores = []
writer = SummaryWriter('PER/reward_per')
noop_max = 30
num_episode = 10000


while True:
  done = False
  state = env.reset()
  while not done:
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    agent.replay_buffer.add((state, action, reward, next_state, done))
    state = next_state
    rand_frames += 1
  print("Random Frame: " + str(rand_frames))
  if rand_frames > 25000:
    break
   

for episode in range(num_episode):
  done = False
  state = env.reset()
  score = 0
  while not done:
    action = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    agent.replay_buffer.add((state, action, reward, next_state, done))
    state = next_state
    score += reward
    agent.learn(num_batch)
    agent.epsilon = max(1 - frames / 100000, agent.epsilon_min)
    frames += 1
    if frames % 1000 == 0:
      print('Update Target Network')
      agent.Q_target.load_state_dict(agent.Q.state_dict())
  scores.append(score)
  avg_score = np.mean(scores[-100:])
  print('episode', episode, 'score %.1f avg score %.1f epsilon %.2f num frames %.1f' % (score, avg_score, agent.epsilon, frames))
  writer.add_scalar('Average Score', avg_score, episode)
  if episode % 100 == 0:
     T.save(agent.Q.state_dict(),'PER/online__per_' + str(episode) + '.t7')
     T.save(agent.Q_target.state_dict(),'PER/target_per_' + str(episode) + '.t7')

writer.close()



