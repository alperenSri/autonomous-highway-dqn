import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo
from moviepy.editor import VideoFileClip, clips_array, concatenate_videoclips
import os
import shutil

ENV_NAME = "highway-fast-v0"
VIDEO_DIR = "videos"
FINAL_VIDEO_NAME = "evolution.mp4"

# --- HIZ VE SÜRE AYARLARI ---
FINAL_DURATION = 30   
SPEED_FACTOR = 4.0   

# Hesaplama: 30sn * 4kat * 15fps = ~1800 adım.
RECORD_STEPS = 2500 

def get_combined_clip(prefix):
    """
    Bir klasördeki tüm parça parça videoları (kaza sonrası oluşanları)
    tek bir uzun video haline getirir.
    """
    temp_dir = os.path.join(VIDEO_DIR, "temp", prefix)
    
    video_files = [f for f in os.listdir(temp_dir) if f.endswith(".mp4")]
    
    if not video_files:
        return None
        
    video_files.sort()
    
    clips = []
    for vid in video_files:
        path = os.path.join(temp_dir, vid)
        clips.append(VideoFileClip(path))
        
    full_clip = concatenate_videoclips(clips, method="compose")
    
    return full_clip

def record_stage(model_path: str, prefix: str, steps: int):
    """
    Simülasyonu kaydeder. Kaza yapınca RESET atar ve KAYDA DEVAM eder.
    """
    temp_dir = os.path.join(VIDEO_DIR, "temp", prefix)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    env = gym.make(ENV_NAME, render_mode='rgb_array')
    
    env.unwrapped.config.update({
        "duration": steps + 100,
        "vehicles_count": 50 
    })
    env.reset()

    env = RecordVideo(
        env, 
        video_folder=temp_dir, 
        episode_trigger=lambda e: True,
        name_prefix=prefix
    )
    
    model = DQN.load(model_path) if model_path else None

    obs, info = env.reset()
    for _ in range(steps):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # SÜREKLİLİK İÇİN: 
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    
    # Parça parça oluşmuş videoları birleştirip döndürür
    return get_combined_clip(prefix)

def create_evolution_video():
    print(f"--- 🎥 Video Kayıt Süreci (Hedef: {FINAL_DURATION}sn @ {SPEED_FACTOR}x Hız) ---")
    print("Not: Araçlar kaza yapınca hemen baştan başlayacak...")
    
    print("Kaydediliyor: Untrained Agent...")
    clip_untrained = record_stage(None, "untrained", steps=RECORD_STEPS)
    
    print("Kaydediliyor: Half-Trained Agent...")
    clip_half = record_stage("models/half_trained", "half", steps=RECORD_STEPS)
    
    print("Kaydediliyor: Fully-Trained Agent...")
    clip_full = record_stage("models/fully_trained", "full", steps=RECORD_STEPS)

    print("--- 🎞️ Videolar İşleniyor ---")

    def process_final(clip):
        if not clip: return None
        
        # Kenar boşluğu
        clip = clip.margin(10)
        
        # HIZLANDIR
        clip = clip.speedx(SPEED_FACTOR)
        
        # SÜREYİ KES
        if clip.duration > FINAL_DURATION:
            return clip.subclip(0, FINAL_DURATION)
        else:
            return clip

    c1 = process_final(clip_untrained)
    c2 = process_final(clip_half)
    c3 = process_final(clip_full)
    
    
    final_clip = clips_array([[c1, c2, c3]])
    
    # 60 FPS Render
    final_clip.write_videofile(FINAL_VIDEO_NAME, fps=60)
    print(f"--- ✨ Video Hazır: {FINAL_VIDEO_NAME} ---")

if __name__ == "__main__":
    create_evolution_video()