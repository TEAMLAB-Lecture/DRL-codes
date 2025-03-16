import os
import random
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import pickle
import imageio
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from pyvirtualdisplay import Display
from huggingface_hub import HfApi, upload_folder

# 환경 변수에서 하이퍼파라미터 가져오기 (기본값 설정)
EPISODES = int(os.environ.get('EPISODES', 20000))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 0.8))
DISCOUNT_FACTOR = float(os.environ.get('DISCOUNT_FACTOR', 0.95))
EPSILON = float(os.environ.get('EPSILON', 1.0))
EPSILON_DECAY = float(os.environ.get('EPSILON_DECAY', 0.999))
MIN_EPSILON = float(os.environ.get('MIN_EPSILON', 0.01))
EVAL_EPISODES = int(os.environ.get('EVAL_EPISODES', 100))
EVAL_INTERVAL = int(os.environ.get('EVAL_INTERVAL', 1000))  # 평가 간격
SLIPPERY = os.environ.get('SLIPPERY', 'True').lower() == 'true'
MAP_SIZE = os.environ.get('MAP_SIZE', '4x4')  # 4x4 또는 8x8
SAVE_ALL_VIDEOS = os.environ.get('SAVE_ALL_VIDEOS', 'True').lower() == 'true'  # 모든 평가 영상 저장 여부

# 출력 디렉토리 생성
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

# 가상 디스플레이 설정 (Docker 환경에서 필요)
display = Display(visible=0, size=(1400, 900))
display.start()

def create_environment():
    """FrozenLake 환경 생성"""
    if MAP_SIZE == '8x8':
        env = gym.make('FrozenLake8x8-v1', render_mode="rgb_array", is_slippery=SLIPPERY)
    else:
        env = gym.make('FrozenLake-v1', render_mode="rgb_array", is_slippery=SLIPPERY)
    return env

def initialize_q_table(state_space, action_space):
    """Q-테이블 초기화"""
    return np.zeros((state_space, action_space))

def epsilon_greedy_policy(q_table, state, epsilon, action_space):
    """입실론-그리디 정책"""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_space - 1)  # 탐색
    else:
        return np.argmax(q_table[state, :])  # 활용

def record_episode(env, q_table):
    """에피소드 실행 및 프레임 기록"""
    frames = []
    state, _ = env.reset()
    frames.append(env.render())
    done = False
    success = False
    
    step_count = 0
    while not done:
        action = np.argmax(q_table[state, :])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # 2스텝마다 프레임 저장 (또는 종료 시)
        if step_count % 2 == 0 or done:
            frames.append(env.render())
        
        if terminated and reward == 1.0:
            success = True
            
        state = next_state
    
    return frames, success

def evaluate_during_training(env, q_table, eval_episodes=10, episode_num=0, should_save_video=False):
    """훈련 중 에이전트 평가"""
    success_count = 0
    all_frames = []
    
    # 첫 번째 에피소드는 항상 기록
    frames, success = record_episode(env, q_table)
    all_frames.extend(frames)
    if success:
        success_count += 1
    
    # 나머지 에피소드 평가
    for _ in range(eval_episodes - 1):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = np.argmax(q_table[state, :])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if terminated and reward == 1.0:
                success_count += 1
                
            state = next_state
    
    success_rate = success_count / eval_episodes * 100
    
    # 영상 저장 - 항상 저장하도록 수정
    if should_save_video:  # 변수명 변경
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(VIDEOS_DIR, f"episode_{episode_num:06d}_{timestamp}.mp4")
        save_video(all_frames, video_path)
        print(f"중간 평가 영상 저장됨: {video_path}")
    
    return success_rate, all_frames

def train_agent(env, q_table, episodes, learning_rate, discount_factor, epsilon, epsilon_decay, min_epsilon, eval_interval):
    """Q-Learning을 사용한 에이전트 훈련"""
    # 학습 과정 기록을 위한 변수
    training_history = {
        'episodes': [],
        'success_rates': [],
        'epsilons': [],
        'avg_rewards': []
    }
    
    best_success_rate = 0
    best_q_table = None
    eval_env = create_environment()  # 평가용 환경 생성
    
    episode_rewards = []
    all_eval_frames = []  # 모든 평가 프레임 저장
    
    for episode in tqdm(range(episodes), desc="Training"):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # 입실론 감소
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        while not done:
            # 행동 선택
            action = epsilon_greedy_policy(q_table, state, epsilon, env.action_space.n)
            
            # 행동 수행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Q-테이블 업데이트 (Q-Learning 알고리즘)
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            
            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[state, action] = new_value
            
            state = next_state
        
        episode_rewards.append(total_reward)
        
        # 일정 간격으로 평가 및 기록
        if (episode + 1) % eval_interval == 0 or episode == episodes - 1:
            success_rate, eval_frames = evaluate_during_training(
                eval_env, q_table, eval_episodes=5, 
                episode_num=episode+1, should_save_video=True
            )
            
            # 평가 프레임 저장
            all_eval_frames.extend(eval_frames)
            
            # 학습 과정 기록
            training_history['episodes'].append(episode + 1)
            training_history['success_rates'].append(success_rate)
            training_history['epsilons'].append(epsilon)
            
            # 최근 100개 에피소드의 평균 보상 계산
            recent_rewards = episode_rewards[-min(100, len(episode_rewards)):]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            training_history['avg_rewards'].append(avg_reward)
            
            print(f"Episode {episode+1}/{episodes}, Success Rate: {success_rate:.2f}%, Epsilon: {epsilon:.4f}, Avg Reward: {avg_reward:.4f}")
            
            # 최적의 모델 저장
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_q_table = q_table.copy()
                
                # 최적 모델 저장
                timestamp_model = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(MODELS_DIR, f"best_q_table_ep{episode+1}_{timestamp_model}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(best_q_table, f)
                print(f"New best model saved with success rate: {best_success_rate:.2f}%")
    
    # 학습 과정 저장
    timestamp_history = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(LOGS_DIR, f"training_history_{timestamp_history}.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    
    # 학습 과정 시각화
    plot_training_history(training_history, timestamp_history)
    
    # 모든 평가 영상을 하나로 합치기 - 항상 합치도록 수정
    if all_eval_frames:  # SAVE_ALL_VIDEOS 조건 제거
        combined_video_path = os.path.join(OUTPUT_DIR, f"learning_progress_{timestamp_history}.mp4")
        save_video(all_eval_frames, combined_video_path)
        print(f"학습 과정 영상이 하나로 합쳐졌습니다: {combined_video_path}")
    
    eval_env.close()
    return best_q_table if best_q_table is not None else q_table, epsilon, training_history

def plot_training_history(history, timestamp=None):
    """학습 과정 시각화"""
    plt.figure(figsize=(15, 10))
    
    # 성공률 그래프
    plt.subplot(3, 1, 1)
    plt.plot(history['episodes'], history['success_rates'])
    plt.title('Success Rate During Training')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (%)')
    plt.grid(True)
    
    # 입실론 그래프
    plt.subplot(3, 1, 2)
    plt.plot(history['episodes'], history['epsilons'])
    plt.title('Epsilon Decay During Training')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    # 평균 보상 그래프
    plt.subplot(3, 1, 3)
    plt.plot(history['episodes'], history['avg_rewards'])
    plt.title('Average Reward During Training')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.tight_layout()
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(LOGS_DIR, f"training_history_{timestamp}.png"))
    plt.close()

def evaluate_agent(env, q_table, episodes):
    """훈련된 에이전트 평가"""
    success = 0
    frames = []
    
    for episode in tqdm(range(episodes), desc="Evaluating"):
        state, _ = env.reset()
        done = False
        
        # 첫 번째 에피소드만 프레임 저장
        save_frames = episode == 0
        if save_frames:
            frames.append(env.render())
        
        step_count = 0
        while not done:
            # 최적 행동 선택 (탐색 없음)
            action = np.argmax(q_table[state, :])
            
            # 행동 수행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
            
            # 프레임 저장 (매 스텝마다 저장하지 않고 2스텝마다 저장)
            if save_frames and (step_count % 2 == 0 or done):
                frames.append(env.render())
            
            if terminated and reward == 1.0:
                success += 1
                
            state = next_state
    
    success_rate = success / episodes * 100
    print(f"Success rate: {success_rate:.2f}%")
    
    return frames, success_rate

def save_video(frames, filename):
    """평가 과정의 비디오 저장"""
    # 동영상 품질 개선을 위한 옵션 설정
    imageio.mimsave(
        filename, 
        frames, 
        fps=8,  # 2fps에서 8fps로 변경 (4배 빠르게)
        quality=8,  # 품질 설정 (0-10, 높을수록 좋음)
        macro_block_size=1  # 더 부드러운 동영상을 위한 설정
    )
    print(f"Video saved to {filename}")

def save_model(q_table, filename):
    """Q-테이블 저장"""
    with open(filename, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"Model saved to {filename}")

def combine_videos(output_path):
    """개별 평가 영상들을 하나로 합치기"""
    video_files = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    if not video_files:
        print("No video files found to combine")
        return
    
    all_frames = []
    for video_file in tqdm(video_files, desc="Combining videos"):
        try:
            reader = imageio.get_reader(video_file)
            for frame in reader:
                all_frames.append(frame)
            reader.close()
        except Exception as e:
            print(f"Error reading {video_file}: {e}")
    
    if all_frames:
        save_video(all_frames, output_path)
        print(f"Combined video saved to {output_path}")
    else:
        print("No frames were extracted from videos")

def main():
    """메인 함수"""
    timestamp_start = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 환경 생성
    env = create_environment()
    print(f"Environment: FrozenLake {'8x8' if MAP_SIZE == '8x8' else '4x4'}")
    print(f"Slippery: {SLIPPERY}")
    print(f"학습 과정 영상이 자동으로 저장됩니다.")
    
    # Q-테이블 초기화
    state_space = env.observation_space.n
    action_space = env.action_space.n
    q_table = initialize_q_table(state_space, action_space)
    
    # 에이전트 훈련
    print(f"Training for {EPISODES} episodes...")
    print(f"Learning rate: {LEARNING_RATE}, Discount factor: {DISCOUNT_FACTOR}")
    print(f"Initial epsilon: {EPSILON}, Epsilon decay: {EPSILON_DECAY}, Min epsilon: {MIN_EPSILON}")
    print(f"Evaluation interval: {EVAL_INTERVAL} episodes")
    
    start_time = time.time()
    q_table, final_epsilon, training_history = train_agent(
        env, q_table, EPISODES, LEARNING_RATE, DISCOUNT_FACTOR, 
        EPSILON, EPSILON_DECAY, MIN_EPSILON, EVAL_INTERVAL
    )
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final epsilon: {final_epsilon:.4f}")
    
    # 최종 모델 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(OUTPUT_DIR, f"final_q_table_{timestamp}.pkl")
    save_model(q_table, model_path)
    
    # 에이전트 평가
    print(f"Evaluating agent for {EVAL_EPISODES} episodes...")
    frames, success_rate = evaluate_agent(env, q_table, EVAL_EPISODES)
    
    # 비디오 저장
    video_path = os.path.join(OUTPUT_DIR, f"frozen_lake_solution_{timestamp}.mp4")
    save_video(frames, video_path)
    
    # 결과 요약
    print("\n===== Training Summary =====")
    print(f"Environment: FrozenLake {'8x8' if MAP_SIZE == '8x8' else '4x4'}")
    print(f"Slippery: {SLIPPERY}")
    print(f"Episodes: {EPISODES}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Discount factor: {DISCOUNT_FACTOR}")
    print(f"Final epsilon: {final_epsilon:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Success rate: {success_rate:.2f}%")
    
    # 최적 모델 정보 출력
    best_success_rate = max(training_history['success_rates']) if training_history['success_rates'] else 0
    best_episode = training_history['episodes'][training_history['success_rates'].index(best_success_rate)] if best_success_rate > 0 else 0
    
    print(f"\n===== Best Model =====")
    print(f"Best success rate: {best_success_rate:.2f}%")
    print(f"Found at episode: {best_episode}")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Final video saved at: {video_path}")
    print(f"Training history saved in: {LOGS_DIR}")
    
    # 개별 평가 영상들을 하나로 합치기 (이미 train_agent에서 합쳐졌을 수 있음)
    if not os.path.exists(os.path.join(OUTPUT_DIR, f"learning_progress_{timestamp_start}.mp4")):
        print("\n개별 평가 영상들을 하나로 합치는 중...")
        combined_video_path = os.path.join(OUTPUT_DIR, f"learning_progress_{timestamp}.mp4")
        combine_videos(combined_video_path)
    
    # 환경 종료
    env.close()

if __name__ == "__main__":
    main() 