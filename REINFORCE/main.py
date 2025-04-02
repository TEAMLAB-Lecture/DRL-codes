import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import os
import matplotlib.pyplot as plt
import imageio
import gym

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 하이퍼파라미터
BATCH_SIZE = 128  # 배치 크기 증가
LEARNING_RATE = 0.001  # 학습률 조정
GAMMA = 0.99
EPISODES = 1000
MAX_STEPS = 1000
RENDER_INTERVAL = 100
SAVE_INTERVAL = 100

class Grid:
    def __init__(self, size=10, start_pos=(0, 0), exit_pos=(6, 5)):
        self.size = size
        self.exit_pos = exit_pos
        self.start_pos = start_pos
        self.figure_pos = start_pos

    def reset(self):
        self.figure_pos = self.start_pos

    def move(self, direction):
        x, y = self.figure_pos
        if direction == 0 and y > 0:  # up
            self.figure_pos = (x, y-1)
        elif direction == 1 and y < self.size-1:  # down
            self.figure_pos = (x, y+1)
        elif direction == 2 and x > 0:  # left
            self.figure_pos = (x-1, y)
        elif direction == 3 and x < self.size-1:  # right
            self.figure_pos = (x+1, y)

    def is_at_exit(self):
        return self.figure_pos == self.exit_pos

    def get_state(self, device='cpu'):
        return torch.FloatTensor(self.figure_pos).unsqueeze(0).to(device)

class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

def generate_episode(grid, policy_net, device=device, max_episode_len=100):
    episode = []
    state = grid.get_state(device)
    ep_length = 0

    while not grid.is_at_exit() and ep_length < max_episode_len:
        ep_length += 1
        action_probs = policy_net(state).squeeze()
        log_probs = torch.log(action_probs)
        cpu_action_probs = action_probs.detach().cpu().numpy()
        action = np.random.choice(np.arange(4), p=cpu_action_probs)

        grid.move(action)
        next_state = grid.get_state(device)
        reward = -0.1 if not grid.is_at_exit() else 1.0

        episode.append((state.tolist(), action, reward, log_probs.tolist()))

        if reward == 1.0:
            break

        state = next_state

    return episode

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    R = 0
    for reward in reversed(rewards):
        R = reward + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards

def train_policy_net(policy_net, optimizer, states, actions, log_probs, discounted_rewards, device=device):
    policy_net.train()
    optimizer.zero_grad()

    # 배치 처리 최적화
    states = torch.cat([torch.tensor(s, dtype=torch.float32) for s in states]).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)

    # 배치 단위로 처리
    action_probs = policy_net(states)
    log_action_probs = torch.log(action_probs)
    selected_log_probs = log_action_probs.gather(1, actions.unsqueeze(1)).squeeze()
    
    # 손실 계산 및 역전파
    loss = -selected_log_probs * discounted_rewards
    loss = loss.mean()
    loss.backward()
    
    # 그래디언트 클리핑 추가
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss.item()

def visualize_episode(grid, policy_net, device=device, max_episode_len=100):
    frames = []
    episode_info = []

    policy_net.eval()
    with torch.no_grad():
        for step in range(max_episode_len):
            plt.figure(figsize=(5, 5))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(np.zeros((grid.size, grid.size)), cmap='gray', vmin=0, vmax=1)
            plt.text(grid.exit_pos[0], grid.exit_pos[1], 'Exit', ha='center', va='center', color='green', fontsize=12)
            plt.text(grid.figure_pos[0], grid.figure_pos[1], 'Agent', ha='center', va='center', color='blue', fontsize=12)
            plt.grid(True)
            plt.title(f"Step: {step + 1}")
            plt.savefig('frame.png')
            plt.close()
            frames.append(imageio.imread('frame.png'))

            state = grid.get_state(device)
            action_probs = policy_net(state).squeeze()
            action = np.random.choice(np.arange(4), p=action_probs.detach().cpu().numpy())
            episode_info.append((state.cpu().numpy().tolist(), action))

            grid.move(action)
            if grid.is_at_exit():
                break

    imageio.mimsave('game_progress.gif', frames, fps=1)
    imageio.mimsave('game_progress.mp4', frames, fps=1)

def visualize_successful_episodes(num_episodes=10):
    state_size = 2
    action_size = 4
    hidden_size = 64
    
    policy_net = PolicyNet(state_size, hidden_size, action_size).to(device)
    policy_net.load_state_dict(torch.load('policy_net.pth'))
    policy_net.eval()
    
    successful_episodes = []
    episode_count = 0
    
    while len(successful_episodes) < num_episodes:
        grid = Grid(size=10, start_pos=(0, 0), exit_pos=(6, 5))
        grid.reset()
        frames = []
        
        with torch.no_grad():
            for step in range(MAX_STEPS):
                plt.figure(figsize=(5, 5))
                plt.xticks([])
                plt.yticks([])
                plt.imshow(np.zeros((grid.size, grid.size)), cmap='gray', vmin=0, vmax=1)
                plt.text(grid.exit_pos[0], grid.exit_pos[1], 'Exit', ha='center', va='center', color='green', fontsize=12)
                plt.text(grid.figure_pos[0], grid.figure_pos[1], 'Agent', ha='center', va='center', color='blue', fontsize=12)
                plt.grid(True)
                plt.title(f"Episode {len(successful_episodes) + 1}, Step: {step + 1}")
                plt.savefig('frame.png')
                plt.close()
                frames.append(imageio.imread('frame.png'))
                
                state = grid.get_state(device)
                action_probs = policy_net(state).squeeze()
                action = np.argmax(action_probs.cpu().numpy())  # 가장 높은 확률의 액션 선택
                
                grid.move(action)
                if grid.is_at_exit():
                    successful_episodes.append(frames)
                    break
                    
        episode_count += 1
        if episode_count > num_episodes * 2:  # 너무 많은 시도를 방지
            break
    
    # 성공적인 에피소드들을 하나의 비디오로 합치기
    all_frames = []
    for i, frames in enumerate(successful_episodes):
        all_frames.extend(frames)
        # 에피소드 사이에 구분선 추가
        for _ in range(5):
            plt.figure(figsize=(5, 5))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(np.zeros((grid.size, grid.size)), cmap='gray', vmin=0, vmax=1)
            plt.text(grid.size/2, grid.size/2, f'Episode {i+1} Complete!', 
                    ha='center', va='center', color='red', fontsize=12)
            plt.savefig('frame.png')
            plt.close()
            all_frames.append(imageio.imread('frame.png'))
    
    imageio.mimsave('successful_episodes.mp4', all_frames, fps=2)
    print(f"성공적인 {len(successful_episodes)}개의 에피소드를 시각화했습니다.")

def train():
    state_size = 2  # x, y 좌표
    action_size = 4  # 상, 하, 좌, 우
    hidden_size = 64  # 은닉층 크기 증가

    policy_net = PolicyNet(state_size, hidden_size, action_size).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    # 학습률 스케줄러 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_reward = -float('inf')
    episode_rewards = []

    for i in range(EPISODES):
        print(f"Iteration {i+1}/{EPISODES}")
        grid = Grid(size=10, start_pos=(0, 0), exit_pos=(6, 5))
        grid.reset()
        episode = generate_episode(grid, policy_net)
        rewards = [step[2] for step in episode]
        discounted_rewards = compute_discounted_rewards(rewards)

        states = [step[0] for step in episode]
        actions = [step[1] for step in episode]
        log_probs = [step[3] for step in episode]

        loss = train_policy_net(policy_net, optimizer, states, actions, log_probs, discounted_rewards)
        print(f"Loss: {loss:.4f}")

        episode_rewards.append(sum(rewards))
        if (i + 1) % RENDER_INTERVAL == 0:
            visualize_episode(grid, policy_net)

        if (i + 1) % SAVE_INTERVAL == 0:
            torch.save(policy_net.state_dict(), 'policy_net.pth')
            print(f"모델 저장 완료 (Iteration {i+1})")

        # 학습률 스케줄러 업데이트
        scheduler.step(sum(rewards))

    print("학습 완료!")
    visualize_episode(grid, policy_net)

if __name__ == "__main__":
    train()
    print("\n성공적인 에피소드 시각화 중...")
    visualize_successful_episodes() 