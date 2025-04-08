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

# 로깅 설정
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'actor_critic_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic:
    def __init__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.episode_losses = []
        logger.info(f"ActorCritic initialized with lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}")
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[action]
        
    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        
        # Get current state value and next state value
        value = self.critic(state)
        next_value = self.critic(next_state) if not done else torch.FloatTensor([0])
        
        # Calculate TD target and TD error
        td_target = reward + self.gamma * next_value
        td_error = td_target - value
        
        # Critic update using MSE loss
        critic_loss = td_error.pow(2)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # Actor update using the original action probability
        probs = self.actor(state)
        action_prob = probs[action]
        actor_loss = -torch.log(action_prob) * td_error.detach()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # Store total loss for logging
        total_loss = (critic_loss + actor_loss).item()
        self.episode_losses.append(total_loss)
        
        return total_loss

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
    
    agent = ActorCritic(state_dim, action_dim)
    rewards_history = []
    
    # Create directories for saving videos and plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = f"videos_{timestamp}"
    os.makedirs(video_dir, exist_ok=True)
    logger.info(f"Video directory created: {video_dir}")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        frames = []
        
        while True:
            # Render and save frame
            frame = env.render()
            frames.append(frame)
            
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            loss = agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Save video every 20 episodes
        if episode % 20 == 0:
            save_episode_video(frames, episode, video_dir)
            logger.info(f"Episode {episode} finished with {len(frames)} steps")
        
        rewards_history.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_loss = np.mean(agent.episode_losses[-10:])
            logger.info(f"Episode {episode}")
            logger.info(f"  Average Reward (last 10): {avg_reward:.2f}")
            logger.info(f"  Average Loss (last 10): {avg_loss:.4f}")
            logger.info(f"  Episode Return: {episode_reward:.2f}")
            
            # Plot and save rewards
            plt.figure(figsize=(10, 5))
            plt.plot(rewards_history)
            plt.title("Training Rewards")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig(os.path.join(video_dir, f"rewards_{timestamp}.png"))
            plt.close()
            
            # Plot and save losses
            plt.figure(figsize=(10, 5))
            plt.plot(agent.episode_losses)
            plt.title("Training Losses")
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.savefig(os.path.join(video_dir, f"losses_{timestamp}.png"))
            plt.close()
    
    env.close()
    logger.info("Training completed")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise 