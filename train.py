import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import os

# Konfigürasyonlar
MODELS_DIR = "models"
LOG_DIR = "logs"
ENV_NAME = "highway-fast-v0"
TOTAL_TIMESTEPS = 20000  # Hızlı sonuç için 

def train_agent():
    """
    Ajanı eğitir ve iki aşamada (yarı ve tam) modelleri kaydeder.
    """
    # Dizinleri oluştur
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Ortamı başlat
    env = gym.make(ENV_NAME, render_mode='rgb_array')

    # Modeli başlat 
    model = DQN(
        "MlpPolicy", 
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=LOG_DIR
    )

    print("--- 🚀 Eğitim Başlıyor... ---")

    # Yarı Eğitim (Half-Trained)
    half_steps = TOTAL_TIMESTEPS // 2
    model.learn(total_timesteps=half_steps, reset_num_timesteps=False)
    model.save(f"{MODELS_DIR}/half_trained")
    print("--- ✅ Yarı Eğitim Tamamlandı ve Kaydedildi ---")

    # Tam Eğitim (Fully-Trained)
    model.learn(total_timesteps=half_steps, reset_num_timesteps=False)
    model.save(f"{MODELS_DIR}/fully_trained")
    print("--- 🏁 Tam Eğitim Tamamlandı ve Kaydedildi ---")
    
    env.close()

if __name__ == "__main__":
    train_agent()