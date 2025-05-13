import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from trpo import TRPO
import os

def evaluate(env_name, agent, seed, eval_iterations, record_video=False):
    if record_video:
        video_dir = os.path.join(os.getcwd(), "videos")
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_dir,
            episode_trigger=lambda x: True,  # 모든 에피소드 녹화
            name_prefix=f"trpo_pendulum_eval"
        )
    else:
        env = gym.make(env_name)
    
    scores = []
    for i in range(eval_iterations):
        (s, _), terminated, truncated, score = env.reset(seed=seed + 100 + i), False, False, 0
        while not (terminated or truncated):
            a = agent.act(s, training=False)
            s_prime, r, terminated, truncated, _ = env.step(2.0 * a)
            score += r
            s = s_prime
        scores.append(score)
    env.close()
    return round(np.mean(scores), 4)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    env_name = 'Pendulum-v1'
    
    # 비디오 저장을 위한 디렉토리 생성
    video_dir = os.path.join(os.getcwd(), "videos")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    seed = 0
    seed_all(seed)
    hidden_dims = (64, 64)
    # 학습 횟수 증가
    max_iterations = 1000000  # 10,000 -> 1,000,000
    eval_intervals = 10000    # 1,000 -> 10,000
    eval_iterations = 10      # 5 -> 10
    batch_size = 2000        # 1,000 -> 2,000
    activation_fn = torch.tanh
    gamma = 0.99             # 0.95 -> 0.99 (더 먼 미래의 보상 고려)
    lmda = 0.95
    backtrack_alpha = 0.5

    env = gym.make(env_name, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TRPO(
        state_dim,
        action_dim,
        hidden_dims=hidden_dims,
        activation_fn=activation_fn,
        batch_size=batch_size,
        gamma=gamma,
        lmda=lmda,
        backtrack_alpha=backtrack_alpha,
    )

    logger = []
    (s, _), terminated, truncated = env.reset(seed=seed), False, False
    for t in tqdm(range(1, max_iterations + 1)):
        a = agent.act(s)
        s_prime, r, terminated, truncated, _ = env.step(2.0 * a)
        result = agent.step((s, a, r, s_prime, terminated))
        s = s_prime
        
        if result is not None:
            logger.append([t, 'policy_loss', result['policy_loss']])
            logger.append([t, 'value_loss', result['value_loss']])
        
        if terminated or truncated:
            (s, _), terminated, truncated = env.reset(), False, False
            
        if t % eval_intervals == 0:
            # 마지막 평가에서는 비디오 녹화
            record_video = (t == max_iterations)
            score = evaluate(env_name, agent, seed, eval_iterations, record_video)
            logger.append([t, 'Avg return', score])
            print(f"Step {t}, Average Return: {score}")

    # 학습 곡선 시각화
    logger = pd.DataFrame(logger)
    logger.columns = ['step', 'key', 'value']

    fig = plt.figure(figsize=(12, 4))

    ax = fig.add_subplot(1, 3, 1)
    key = 'Avg return'
    ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
    ax.grid(axis='y')
    ax.set_title("Average return over 10 episodes")
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg return")

    ax = fig.add_subplot(1, 3, 2)
    key = 'policy_loss'
    ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
    ax.grid(axis='y')
    ax.set_title("Policy loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Policy loss")

    ax = fig.add_subplot(1, 3, 3)
    key = 'value_loss'
    ax.plot(logger.loc[logger['key'] == key, 'step'], logger.loc[logger['key'] == key, 'value'], 'b-')
    ax.grid(axis='y')
    ax.set_title("Value loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value loss")

    fig.tight_layout()
    plt.savefig('trpo_pendulum_learning_curves.png')
    plt.close() 