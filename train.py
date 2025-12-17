import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import pandas as pd
import glob, os

print(">>> train.py is running <<<")

# -----------------------------
# 1. Định nghĩa môi trường Microgrid
# -----------------------------
class MicrogridEnv(gym.Env):
    def __init__(self, pv_profile, load_profile):
        super(MicrogridEnv, self).__init__()
        self.pv_profile = pv_profile
        self.load_profile = load_profile
        self.t = 0
        self.SoC = 0.5

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.SoC = 0.5
        obs = np.array([self.SoC, self.pv_profile[self.t], self.load_profile[self.t], 0.05], dtype=np.float32)
        return obs, {}

    def step(self, action):
        charge_action, curtail_action = action
        self.SoC = np.clip(self.SoC + 0.05 * float(charge_action), 0.0, 1.0)

        PV_out = self.pv_profile[self.t] * (1.0 - float(curtail_action))
        grid_import = self.load_profile[self.t] - PV_out - float(charge_action) * 0.5

        cost = grid_import * 0.15
        penalty = 0.1 * float(curtail_action) + 0.2 * abs(self.SoC - 0.5)
        reward = - (cost + penalty)

        self.t = (self.t + 1) % 24
        obs = np.array([self.SoC, self.pv_profile[self.t], self.load_profile[self.t], 0.05], dtype=np.float32)

        terminated = (self.t == 0)
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

# -----------------------------
# 2. Dữ liệu PV & Load mẫu (24h)
# -----------------------------
pv_profile = [
    0.0, 0.0, 0.0, 0.0, 0.05, 0.15, 0.35, 0.55,
    0.75, 0.90, 0.95, 1.0, 0.95, 0.85, 0.65, 0.40,
    0.20, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]
load_profile = [
    0.6, 0.55, 0.50, 0.50, 0.55, 0.70, 0.90, 1.2,
    1.5, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0, 1.2,
    1.5, 1.8, 2.0, 2.2, 2.0, 1.5, 1.0, 0.8
]

# -----------------------------
# 3. Huấn luyện PPO
# -----------------------------
env = DummyVecEnv([lambda: MicrogridEnv(pv_profile, load_profile)])
model = PPO("MlpPolicy", env, verbose=1)

timesteps = 100000  # chỉnh số bước train ở đây
print(f">>> Bắt đầu huấn luyện PPO với {timesteps} steps...")
model.learn(total_timesteps=timesteps)
print(">>> Huấn luyện xong, bắt đầu test...")

# -----------------------------
# 4. Test agent đã học + lưu dữ liệu
# -----------------------------
obs = env.reset()
soc_list, pv_list, load_list, reward_list = [], [], [], []

for t in range(24):
    action, _ = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)

    soc_list.append(obs[0][0])
    pv_list.append(obs[0][1])
    load_list.append(obs[0][2])
    reward_list.append(rewards[0])

    print(f"Hour={t:02d}, SoC={obs[0][0]:.2f}, PV={obs[0][1]:.2f}, Load={obs[0][2]:.2f}, Reward={rewards[0]:.3f}")
    if dones[0]:
        break

# -----------------------------
# 5. Xuất CSV (tên file theo số bước train)
# -----------------------------
csv_name = f"results_{timesteps//1000}k.csv"
df = pd.DataFrame({
    "Hour": list(range(len(soc_list))),
    "SoC": soc_list,
    "PV": pv_list,
    "Load": load_list,
    "Reward": reward_list
})
df.to_csv(csv_name, index=False, encoding="utf-8-sig")
print(f">>> Đã lưu kết quả test vào {csv_name}")

# -----------------------------
# 6. Vẽ biểu đồ kết quả hiện tại
# -----------------------------
hours = list(range(len(soc_list)))
plt.figure(figsize=(10,6))
plt.plot(hours, soc_list, label="SoC", marker="o")
plt.plot(hours, pv_list, label="PV", marker="s")
plt.plot(hours, load_list, label="Load", marker="^")
plt.plot(hours, reward_list, label="Reward", marker="x")
plt.legend()
plt.xlabel("Hour")
plt.ylabel("Value")
plt.title(f"Microgrid Simulation Results ({timesteps} steps)")
plt.grid(True)
plt.show()

# -----------------------------
# 7. So sánh nhiều file CSV nếu có
# -----------------------------
csv_files = glob.glob("results_*.csv")
if len(csv_files) > 1:
    print(">>> Đang vẽ biểu đồ so sánh nhiều lần train...")
    # SoC comparison
    plt.figure(figsize=(10,6))
    for fname in csv_files:
        label = os.path.splitext(os.path.basename(fname))[0]
        df = pd.read_csv(fname)
        plt.plot(df["Hour"], df["SoC"], label=f"SoC {label}")
    plt.xlabel("Hour"); plt.ylabel("SoC")
    plt.title("SoC Comparison across training runs")
    plt.legend(); plt.grid(True); plt.show()

    # Reward comparison
    plt.figure(figsize=(10,6))
    for fname in csv_files:
        label = os.path.splitext(os.path.basename(fname))[0]
        df = pd.read_csv(fname)
        plt.plot(df["Hour"], df["Reward"], label=f"Reward {label}")
    plt.xlabel("Hour"); plt.ylabel("Reward")
    plt.title("Reward Comparison across training runs")
    plt.legend(); plt.grid(True); plt.show()