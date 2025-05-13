import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import os
import logging
from collections import deque

# 로깅 설정
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class NStepActorCritic:
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.0003, gamma=0.99, n_steps=5):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.n_steps = n_steps
        self.episode_losses = []
        
        # n-step 경험을 저장할 버퍼
        self.state_buffer = deque(maxlen=n_steps)
        self.action_buffer = deque(maxlen=n_steps)
        self.reward_buffer = deque(maxlen=n_steps)
        self.next_state_buffer = deque(maxlen=n_steps)
        self.done_buffer = deque(maxlen=n_steps)
        self.prob_buffer = deque(maxlen=n_steps)
        
        logger.info(f"NStepActorCritic initialized with n_steps={n_steps}, lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}")
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[action]
    
    def store_transition(self, state, action, reward, next_state, done, action_prob):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_state_buffer.append(next_state)
        self.done_buffer.append(done)
        self.prob_buffer.append(action_prob)
    
    def compute_n_step_returns(self):
        returns = []
        if len(self.reward_buffer) == 0:
            return returns
        
        last_state = self.next_state_buffer[-1]
        last_done = self.done_buffer[-1]
        
        if last_done:
            R = 0
        else:
            with torch.no_grad():
                R = self.critic(torch.FloatTensor(last_state)).item()
        
        for i in reversed(range(len(self.reward_buffer))):
            R = self.reward_buffer[i] + self.gamma * R * (not self.done_buffer[i])
            returns.insert(0, R)
        
        return returns
    
    def update(self):
        if len(self.state_buffer) < self.n_steps:
            return 0
        
        returns = self.compute_n_step_returns()
        critic_losses = []
        actor_losses = []
        
        for i in range(len(self.state_buffer)):
            state = torch.FloatTensor(self.state_buffer[i])
            action = self.action_buffer[i]
            R = returns[i]
            
            value = self.critic(state)
            td_error = torch.FloatTensor([R]) - value
            
            critic_loss = td_error.pow(2)
            critic_losses.append(critic_loss.item())
            
            action_prob = self.prob_buffer[i]
            actor_loss = -torch.log(action_prob) * td_error.detach()
            actor_losses.append(actor_loss.item())
        
        critic_loss = torch.mean(torch.stack([torch.FloatTensor([loss]) for loss in critic_losses]))
        actor_loss = torch.mean(torch.stack([torch.FloatTensor([loss]) for loss in actor_losses]))
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        total_loss = (critic_loss + actor_loss).item()
        self.episode_losses.append(total_loss)
        
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.next_state_buffer.clear()
        self.done_buffer.clear()
        self.prob_buffer.clear()
        
        return total_loss

def save_episode_video(frames, episode, video_dir):
    height, width, _ = frames[0].shape
    video_path = os.path.join(video_dir, f"episode_{episode}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
    
    for frame in frames:
        video.write(frame)
    video.release()

def train(env_name="LunarLander-v2", num_episodes=1000, n_steps=5):
    logger.info(f"Starting training with {env_name} for {num_episodes} episodes using {n_steps}-step returns")
    env = gym.make(env_name, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = NStepActorCritic(state_dim, action_dim, n_steps=n_steps)
    rewards_history = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = f"videos_{timestamp}"
    os.makedirs(video_dir, exist_ok=True)
    logger.info(f"Video directory created: {video_dir}")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        frames = []
        steps_since_update = 0
        
        while True:
            frame = env.render()
            frames.append(frame)
            
            action, action_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done, action_prob)
            steps_since_update += 1
            
            if steps_since_update >= n_steps or done:
                loss = agent.update()
                steps_since_update = 0
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        if episode % 20 == 0:
            save_episode_video(frames, episode, video_dir)
            logger.info(f"Episode {episode} finished with {len(frames)} steps")
        
        rewards_history.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_loss = np.mean(agent.episode_losses[-10:]) if agent.episode_losses else 0
            logger.info(f"Episode {episode}")
            logger.info(f"  Average Reward (last 10): {avg_reward:.2f}")
            logger.info(f"  Average Loss (last 10): {avg_loss:.4f}")
            logger.info(f"  Episode Return: {episode_reward:.2f}")
            
            plt.figure(figsize=(10, 5))
            plt.plot(rewards_history)
            plt.title("Training Progress")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.savefig(f"rewards_{timestamp}.png")
            plt.close()
    
    env.close()
    logger.info("Training completed")

if __name__ == "__main__":
    try:
        train(n_steps=5)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise 