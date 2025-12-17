import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandapower as pp
import pandapower.networks as nw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# 1. Định nghĩa môi trường IEEE-33
# -----------------------------
class IEEE33Env(gym.Env):
    def __init__(self):
        super(IEEE33Env, self).__init__()

        # Load lưới IEEE-33 bus chuẩn
        self.net = nw.case33bw()

        # Pin đặt tại bus 18 (index = 17)
        self.soc = 0.5
        self.max_power = 0.2   # MW

        # Action: [-1,1] => xả/sạc pin
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: SoC + điện áp tại 5 bus quan trọng
        self.obs_buses = [0, 9, 17, 24, 32]
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(1+len(self.obs_buses),), dtype=np.float32)

        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc = 0.5
        self.t = 0

        # Chạy power flow lần đầu
        pp.runpp(self.net, numba=False)

        obs = self._get_obs()
        return obs, {}   # Gymnasium chuẩn: (obs, info)
    

    def step(self, action):
        act = float(action[0])
        delta_p = act * self.max_power
        self.soc = np.clip(self.soc + act*0.05, 0.0, 1.0)

        # Xóa BESS cũ nếu có
        bess_idx = self.net.sgen[self.net.sgen.name == "BESS"].index
        if len(bess_idx) > 0:
            self.net.sgen.drop(bess_idx, inplace=True)

        # Thêm BESS mới
        pp.create_sgen(self.net, bus=17, p_mw=delta_p, q_mvar=0.0, name="BESS")

        # Chạy power flow
        pp.runpp(self.net, numba=False)

        # Reward: âm tổn thất + penalty điện áp
        loss = self.net.res_line.pl_mw.sum()
        voltages = self.net.res_bus.vm_pu.values
        penalty = np.sum(np.maximum(0, np.abs(voltages-1.0)-0.05))
        reward = -loss - penalty

        obs = self._get_obs()
        self.t += 1
        terminated = (self.t >= 24)
        truncated = False
        info = {"loss": loss}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        if self.net.res_bus.empty:
            pp.runpp(self.net, numba=False)
        voltages = self.net.res_bus.vm_pu.values
        obs = [self.soc] + [voltages[i] for i in self.obs_buses]
        return np.array(obs, dtype=np.float32)

# -----------------------------
# 2. Train PPO
# -----------------------------
env = DummyVecEnv([lambda: IEEE33Env()])
model = PPO("MlpPolicy", env, verbose=1)

timesteps = 2000   # test nhanh, có thể tăng lên 50k
print(f"=> Bắt đầu train với {timesteps} steps")
model.learn(total_timesteps=timesteps)
print("=> Train xong, bắt đầu test...")

# -----------------------------
# -----------------------------
# 3. Test agent đã học (VecEnv)
# -----------------------------
obs = env.reset()  # VecEnv reset chỉ trả về obs
soc_list, reward_list, loss_list = [], [], []

for t in range(24):
    action, _ = model.predict(obs)

    # SB3 VecEnv có 2 dạng trả về tùy phiên bản wrapper:
    result = env.step(action)
    if len(result) == 4:
        # Dạng cũ: (obs, rewards, dones, infos)
        obs, rewards, dones, infos = result
        done = dones[0]
        loss = infos[0].get("loss")
        reward_val = rewards[0]
    else:
        # Dạng Gymnasium: (obs, rewards, terminated, truncated, infos)
        obs, rewards, terminated, truncated, infos = result
        done = (terminated[0] or truncated[0])
        loss = infos[0].get("loss")
        reward_val = rewards[0]

    # Ghi log
    soc_list.append(obs[0][0])
    reward_list.append(reward_val)
    loss_list.append(loss)

    print(f"Hour={t}, SoC={obs[0][0]:.2f}, Reward={reward_val:.3f}, Loss={loss:.3f}")

    if done:
        break

# -----------------------------
# 4. Xuất CSV
# -----------------------------
df = pd.DataFrame({
    "Hour": list(range(len(soc_list))),
    "SoC": soc_list,
    "Reward": reward_list,
    "Loss": loss_list
})
df.to_csv(f"ieee33_results_{timesteps//1000}k.csv", index=False, encoding="utf-8-sig")
print("=> Đã lưu kết quả test ra CSV")

# -----------------------------
# 5. Vẽ biểu đồ
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(df["Hour"], df["SoC"], label="SoC", marker="o")
plt.plot(df["Hour"], df["Reward"], label="Reward", marker="x")
plt.plot(df["Hour"], df["Loss"], label="Loss", marker="s")
plt.xlabel("Hour")
plt.ylabel("Value")
plt.title(f"IEEE-33 PPO Results ({timesteps} steps)")
plt.legend()
plt.grid(True)
plt.show()