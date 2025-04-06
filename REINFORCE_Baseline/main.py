import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import os
import logging
from datetime import datetime

# 로깅 설정
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'reinforce_baseline_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        logger.info(f"Policy Network initialized with state_dim={state_dim}, action_dim={action_dim}")
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class REINFORCEBaseline:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.episode_returns = []
        logger.info(f"REINFORCEBaseline initialized with lr={lr}, gamma={gamma}")
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy_net(state)
        action = torch.multinomial(probs, 1).item()
        return action
    
    def update(self, states, actions, rewards):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        
        # 현재 에피소드의 리턴 계산
        G = 0
        returns = []
        for r in rewards[::-1]:
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # 현재까지의 평균 리턴을 baseline으로 사용
        episode_return = sum(rewards)
        self.episode_returns.append(episode_return)
        baseline = np.mean(self.episode_returns) if len(self.episode_returns) > 0 else 0
        
        # 정책 업데이트
        probs = self.policy_net(states)
        selected_probs = probs.gather(1, actions.unsqueeze(1))
        advantages = returns - baseline
        
        policy_loss = -(torch.log(selected_probs) * advantages).mean()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return episode_return, baseline, policy_loss.item()

def save_episode_video(frames, episode, video_dir):
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    video_path = os.path.join(video_dir, f"episode_{episode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    logger.info(f"Episode {episode} video saved: {video_path}")

def train(env_name="CartPole-v1", num_episodes=1000):
    logger.info(f"Starting training with {env_name} for {num_episodes} episodes")
    env = gym.make(env_name, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEBaseline(state_dim, action_dim)
    episode_rewards = []
    
    # 비디오 저장 디렉토리 설정
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)
    logger.info(f"Video directory created: {video_dir}")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        states = []
        actions = []
        rewards = []
        frames = []
        
        while not done:
            frame = env.render()
            frames.append(frame)
            states.append(state)
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            actions.append(action)
            rewards.append(reward)
        
        # 20 에피소드마다 비디오 저장
        if done and episode % 20 == 0:  # 에피소드가 끝나고 20의 배수일 때만 저장
            save_episode_video(frames, episode, video_dir)
            logger.info(f"Episode {episode} finished with {len(frames)} steps")
        
        episode_return, baseline, loss = agent.update(states, actions, rewards)
        episode_rewards.append(episode_return)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode}")
            logger.info(f"  Average Reward (last 10): {avg_reward:.2f}")
            logger.info(f"  Current Baseline: {baseline:.2f}")
            logger.info(f"  Policy Loss: {loss:.4f}")
            logger.info(f"  Episode Return: {episode_return:.2f}")
    
    env.close()
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plot_path = os.path.join(video_dir, f"rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Reward plot saved: {plot_path}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise 