# a3c_pong.py 상단
import os
import random
import numpy as np
import gymnasium as gym
# from gymnasium.wrappers import AtariPreprocessing, FrameStack # 이 방식 대신 아래 방식 사용 권장
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
import ale_py  # <--- 이 줄을 추가하세요!

from tqdm import tqdm
import pickle
import imageio
import time
import json
from datetime import datetime
import glob
from pyvirtualdisplay import Display

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from collections import deque
import cv2

# --- 설정값 ---
MAX_GLOBAL_STEPS = 200000  # 총 학습 스텝 수
RECORD_INTERVAL_GLOBAL_STEPS = 50000  # 영상 녹화 간격
N_WORKERS = 4  # 워커 수
ENV_NAME = 'ALE/Pong-v5'  # 환경 이름 수정

# 출력 디렉토리 생성
OUTPUT_DIR = "output_a3c"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

# 가상 디스플레이 설정
try:
    display = Display(visible=0, size=(400, 300))
    display.start()
    print("가상 디스플레이가 시작되었습니다.")
except Exception as e:
    print(f"가상 디스플레이를 시작할 수 없습니다: {e}. GUI가 필요할 수 있습니다.")

def create_env(render_mode=None):
    """환경 생성 함수"""
    # 원본 환경의 프레임 스킵 비활성화
    env = gym.make(ENV_NAME, render_mode=render_mode, frameskip=1)
    
    # AtariPreprocessing 래퍼가 프레임 스킵을 담당
    env = AtariPreprocessing(env, 
                           frame_skip=4,
                           grayscale_obs=True,
                           scale_obs=True,
                           terminal_on_life_loss=True)
    
    # Gymnasium v1.0.0 이상에서는 FrameStackObservation 사용
    env = FrameStackObservation(env, stack_size=4)
    return env

def save_video(frames, filename, fps=15):
    """평가 과정의 비디오 저장"""
    try:
        imageio.mimsave(
            filename,
            frames,
            fps=fps,
            quality=8,
            macro_block_size=1
        )
        print(f"비디오가 저장되었습니다: {filename}")
    except Exception as e:
        print(f"비디오 저장 중 오류 발생 {filename}: {e}")

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(ActorCriticNetwork, self).__init__()

        # 공유 특징 추출 레이어
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 완전 연결 레이어를 위한 크기 계산
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        self.conv_output_size = self._get_conv_output(dummy_input)

        # 공유 FC 레이어
        self.fc_shared = nn.Linear(self.conv_output_size, 512)

        # 정책 헤드 (Actor)
        self.policy = nn.Linear(512, num_actions)

        # 가치 헤드 (Critic)
        self.value = nn.Linear(512, 1)

    def _get_conv_output(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        # 공유 특징 추출 네트워크
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        shared_features = F.relu(self.fc_shared(x))

        # 정책 헤드: 행동 확률
        policy = F.softmax(self.policy(shared_features), dim=1)

        # 가치 헤드: 상태 가치
        value = self.value(shared_features)

        return policy, value

def preprocess_frame(frame):
    """프레임 전처리"""
    if frame is None:
        return np.zeros((84, 84), dtype=np.float32)
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame

class A3CWorker(mp.Process):
    def __init__(self, global_network, global_optimizer, global_counter, worker_id,
                 video_dir, record_interval_global_steps,
                 n_steps=20, gamma=0.99, entropy_beta=0.01, max_global_steps=MAX_GLOBAL_STEPS):
        super(A3CWorker, self).__init__()

        self.worker_id = worker_id
        self.global_network = global_network
        self.global_optimizer = global_optimizer
        self.global_counter = global_counter
        self.max_global_steps = max_global_steps

        # 영상 기록 관련
        self.video_dir = video_dir
        self.record_interval_global_steps = record_interval_global_steps
        self.last_recorded_step_milestone = 0

        # 하이퍼파라미터
        self.n_steps = n_steps
        self.gamma = gamma
        self.entropy_beta = entropy_beta

        # 환경 생성
        if self.worker_id == 0:
            self.env = create_env(render_mode='rgb_array')
            print(f"워커 {self.worker_id}: render_mode='rgb_array'로 환경 생성")
        else:
            self.env = create_env()
        
        self.input_channels = 4
        num_actions = self.env.action_space.n

        # 로컬 네트워크
        self.local_network = ActorCriticNetwork(self.input_channels, num_actions)
        self.episode_count_worker = 0

    def get_stacked_frames(self, stacked_observation):
        """스택된 관찰을 PyTorch 텐서로 변환"""
        return torch.FloatTensor(np.array(stacked_observation, dtype=np.float32)).unsqueeze(0)

    def sync_with_global(self):
        self.local_network.load_state_dict(self.global_network.state_dict())

    def calculate_loss(self, rewards, values, log_probs, entropies, done_flag):
        R = torch.zeros(1, 1)
        if not done_flag and values:
            R = values[-1].detach()

        policy_loss = 0
        value_loss = 0
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R
            advantage = R - values[i]
            value_loss += 0.5 * advantage.pow(2)
            policy_loss += -(log_probs[i] * advantage.detach() + self.entropy_beta * entropies[i])
        
        total_loss = policy_loss + value_loss
        return total_loss

    def run(self):
        total_episodes_processed_by_worker = 0
        while self.global_counter.value < self.max_global_steps:
            self.sync_with_global()

            log_probs = []
            values_list = []
            rewards = []
            entropies = []

            obs, info = self.env.reset()
            current_stacked_frames = self.get_stacked_frames(obs)

            done_flag = False
            episode_reward = 0
            
            should_record_this_episode = False
            current_episode_raw_frames = []

            if self.worker_id == 0:
                if self.global_counter.value >= self.last_recorded_step_milestone + self.record_interval_global_steps:
                    should_record_this_episode = True
                    self.last_recorded_step_milestone += self.record_interval_global_steps
                    print(f"워커 0: 글로벌 스텝 {self.global_counter.value}에서 에피소드 녹화 시작")
            
            if should_record_this_episode:
                rendered_frame = self.env.render()
                if rendered_frame is not None:
                    current_episode_raw_frames.append(rendered_frame)

            for t_step in range(self.n_steps):
                if done_flag:
                    break

                policy, value = self.local_network(current_stacked_frames)

                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample()

                log_prob = action_dist.log_prob(action)
                entropy = action_dist.entropy()

                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                done_flag = terminated or truncated

                if should_record_this_episode:
                    rendered_frame = self.env.render()
                    if rendered_frame is not None:
                        current_episode_raw_frames.append(rendered_frame)
                
                episode_reward += reward

                log_probs.append(log_prob)
                values_list.append(value)
                rewards.append(torch.FloatTensor([reward]))
                entropies.append(entropy)
                
                current_stacked_frames = self.get_stacked_frames(next_obs)

                with self.global_counter.get_lock():
                    self.global_counter.value += 1
                
                if done_flag:
                    break
            
            if log_probs:
                loss = self.calculate_loss(rewards, values_list, log_probs, entropies, done_flag)

                self.global_optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), 40)
                
                for local_param, global_param in zip(self.local_network.parameters(),
                                                     self.global_network.parameters()):
                    if local_param.grad is not None:
                        global_param._grad = local_param.grad

                self.global_optimizer.step()

            if done_flag:
                total_episodes_processed_by_worker += 1
                print(f"워커 {self.worker_id}, 글로벌 스텝: {self.global_counter.value}, 에피소드: {total_episodes_processed_by_worker}, 보상: {episode_reward:.2f}")

                if should_record_this_episode and current_episode_raw_frames:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = os.path.join(self.video_dir,
                                                  f"pong_worker{self.worker_id}_ep{total_episodes_processed_by_worker}_globStep{self.global_counter.value}_{timestamp}.mp4")
                    save_video(current_episode_raw_frames, video_filename)
                    current_episode_raw_frames = []
                    should_record_this_episode = False

        self.env.close()

def main_a3c():
    start_time = time.time()
    
    input_channels = 4
    temp_env = create_env()
    num_actions = temp_env.action_space.n
    temp_env.close()

    global_network = ActorCriticNetwork(input_channels, num_actions)
    global_network.share_memory()

    global_optimizer = optim.Adam(global_network.parameters(), lr=1e-4)

    global_counter = mp.Value('i', 0)

    workers = []
    for worker_id in range(N_WORKERS):
        worker = A3CWorker(global_network, global_optimizer, global_counter, worker_id,
                            VIDEOS_DIR, RECORD_INTERVAL_GLOBAL_STEPS,
                            max_global_steps=MAX_GLOBAL_STEPS)
        workers.append(worker)

    print(f"{N_WORKERS}개의 워커를 시작합니다...")
    for worker in workers:
        worker.start()

    last_saved_step = 0
    save_model_interval = 100000
    
    monitoring_active = True
    try:
        while monitoring_active:
            time.sleep(20)
            current_steps = global_counter.value
            elapsed_time = time.time() - start_time
            print(f"글로벌 스텝: {current_steps}/{MAX_GLOBAL_STEPS}, 경과 시간: {elapsed_time:.2f}초")

            if current_steps >= last_saved_step + save_model_interval:
                save_path = os.path.join(MODELS_DIR, f"a3c_pong_globalstep_{current_steps}.pt")
                torch.save(global_network.state_dict(), save_path)
                print(f"모델이 저장되었습니다: {save_path}")
                last_saved_step = current_steps

            if current_steps >= MAX_GLOBAL_STEPS:
                print("최대 글로벌 스텝에 도달했습니다. 모니터링을 중지합니다.")
                monitoring_active = False
                break
            
            all_done = True
            for w in workers:
                if w.is_alive():
                    all_done = False
                    break
            if all_done and current_steps < MAX_GLOBAL_STEPS:
                print("모든 워커가 조기에 종료되었습니다.")
                monitoring_active = False

    except KeyboardInterrupt:
        print("사용자에 의해 중단되었습니다. 워커를 종료합니다...")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
                if worker.is_alive():
                    print(f"워커 {worker.worker_id}가 정상적으로 종료되지 않았습니다.")
        print("모든 워커가 종료되었습니다.")

    final_model_path = os.path.join(MODELS_DIR, f"a3c_pong_final_globalstep_{global_counter.value}.pt")
    torch.save(global_network.state_dict(), final_model_path)
    print(f"최종 모델이 저장되었습니다: {final_model_path}")

    end_time = time.time()
    print(f"총 학습 시간: {(end_time - start_time):.2f}초")
    
    if 'display' in globals() and display.is_started:
        display.stop()
        print("가상 디스플레이가 중지되었습니다.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main_a3c() 