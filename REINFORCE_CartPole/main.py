import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size):
        """
        정책 네트워크 초기화
        가이드라인:
        1. 입력 크기: state_size (CartPole의 경우 4)
        2. 출력 크기: action_size (CartPole의 경우 2)
        3. 중간 레이어: 2-3개의 fully connected layer 사용
        4. 활성화 함수: ReLU 사용
        5. 출력층: softmax 사용 (행동 확률 분포 생성)
        """
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        """
        순전파 함수
        가이드라인:
        1. 입력 x를 각 레이어를 통과시킴
        2. 마지막 레이어에서 softmax 적용
        3. 각 행동에 대한 확률 분포 반환
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

class REINFORCE:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.policy_net = PolicyNet(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.gamma = 0.99
        self.memory = deque()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def store_transition(self, state, action, reward, log_prob):
        self.memory.append((state, action, reward, log_prob))

    def train(self):
        """
        REINFORCE 알고리즘의 학습 함수
        가이드라인:
        1. 메모리가 비어있으면 0 반환
        
        2. 할인된 보상 계산:
           - 에피소드의 모든 보상 추출
           - gamma를 사용하여 할인된 누적 보상 계산
           - 보상을 텐서로 변환
        
        3. 정책 손실 계산:
           - 각 상태-행동 쌍에 대해 log_prob * discounted_reward 계산
           - 모든 손실을 합산
        
        4. 정책 업데이트:
           - optimizer.zero_grad()로 그래디언트 초기화
           - loss.backward()로 그래디언트 계산
           - optimizer.step()으로 가중치 업데이트
        
        5. 메모리 초기화
        
        6. 손실값 반환
        """
        if not self.memory:
            return 0

        # 에피소드의 모든 보상 계산
        rewards = [r for _, _, r, _ in self.memory]
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)

        # 정책 손실 계산
        policy_loss = []
        for (_, _, _, log_prob), R in zip(self.memory, discounted_rewards):
            policy_loss.append(-log_prob * R)
        policy_loss = torch.stack(policy_loss).sum()

        # 정책 업데이트
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # 메모리 초기화
        self.memory.clear()
        return policy_loss.item()

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('cartpole_rewards.png')
    plt.close()

def main():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCE(state_size, action_size)
    episodes = 1000
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, log_prob)
            episode_reward += reward
            state = next_state

            if done:
                break

        # 에피소드가 끝나면 학습
        loss = agent.train()
        rewards_history.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}, Loss: {loss:.4f}")

            # 학습된 모델 저장
            if avg_reward > 195.0:  # CartPole-v1의 해결 기준
                torch.save(agent.policy_net.state_dict(), 'cartpole_model.pth')
                print("모델이 성공적으로 저장되었습니다!")
                break

    env.close()
    plot_rewards(rewards_history)

if __name__ == "__main__":
    main() 