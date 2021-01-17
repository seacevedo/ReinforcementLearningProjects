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


class ConvNet(nn.Module):
  def __init__(self, num_actions, input_dims):
    super(ConvNet, self).__init__()

    self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
    self.flat = nn.Flatten()
    self.fc1 = nn.Linear(3136, 512)
    self.fc2 = nn.Linear(512, num_actions)

    self.optimizer = optim.RMSprop(self.parameters(), lr=0.0001)
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

class Agent():
  def __init__(self, num_actions, input_dims, gamma=0.99, epsilon=1.0, epsilon_min=0.01):
    self.gamma = gamma
    self.num_actions = num_actions
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.input_dims = input_dims

    self.Q = ConvNet(self.num_actions, self.input_dims)
    self.Q_target = copy.deepcopy(self.Q)

    self.replay_buffer = deque(maxlen=50000)

  def choose_action(self, observation):
    if np.random.random() < self.epsilon:
      action = np.random.choice(self.num_actions)
    else:
      state = T.tensor(observation, dtype=T.float).to(self.Q.device) / 255
      state = state.unsqueeze(0)
      actions = self.Q.forward(state)
      action = T.argmax(actions).item()
    return action

  def sample_replay(self, num_batch):
    indices = np.random.randint(len(self.replay_buffer),  size=num_batch)
    batch = [self.replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
    return states, actions, rewards, next_states, dones

  def learn(self, num_batch):
    if len(self.replay_buffer) < num_batch:
      return
    self.Q.optimizer.zero_grad()
    states, actions, rewards, next_states, dones = self.sample_replay(num_batch)
    states = T.tensor(states, dtype=T.float).to(self.Q.device) / 255
    actions = T.LongTensor(actions).to(self.Q.device)
    rewards = T.tensor(rewards).to(self.Q.device)
    next_states = T.tensor(next_states, dtype=T.float).to(self.Q.device) / 255
    dones = T.LongTensor(dones).to(self.Q.device)

    q_pred = self.Q.forward(states)[range(states.shape[0]), actions]
    q_next = self.Q_target.forward(next_states).max(dim=1)[0]

    q_target = (rewards + (1 - dones) * self.gamma * q_next)
    loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
    loss.backward()
    self.Q.optimizer.step()

env = gym.make('BreakoutNoFrameskip-v4')
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
agent = Agent(env.action_space.n, env.observation_space.shape)
num_batch = 32
frames = 0
num_episode = 0
scores = []
writer = SummaryWriter('Python Scripts/reward')


while True:
  done = False
  state = env.reset()
  score = 0
  torch.save(agent.Q.state_dict(), 'Python Scripts/online.t7')
  torch.save(agent.Q_target.state_dict(), 'Python Scripts/target.t7')
  break
  while not done:
    action = agent.choose_action(state)
    next_state, reward, done, info = env.step(action)
    agent.replay_buffer.append((state, action, reward, next_state, done))
    state = next_state
    score += reward
    agent.learn(num_batch)
    agent.epsilon = max(1 - frames / 100000, agent.epsilon_min)
    frames += 1
    if frames % 1000 == 0:
      print('Update Target Network')
      agent.Q_target.load_state_dict(agent.Q.state_dict())
  scores.append(score)
  num_episode += 1
  avg_score = np.mean(scores[-100:])
  print('episode', num_episode, 'score %.1f avg score %.1f epsilon %.2f num frames %.1f' % (score, avg_score, agent.epsilon, frames))
  writer.add_scalar('Average Score', avg_score, num_episode)
  if frames >= 1000000:
    break

writer.close()