import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandapower as pp
import pandapower.networks as nw
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# -----------------------------
# 1) Môi trường có BESS + profile tải + PV/Wind
# -----------------------------
class IEEE33EnvConvergedRES(gym.Env):
    def __init__(self):
        super().__init__()
        self.net = nw.case33bw()

        # BESS
        self.soc = 0.5
        self.max_power = 0.2     # MW
        self.soc_step = 0.06
        self.soc_low, self.soc_high = 0.2, 0.9

        # Quan sát và hành động
        self.obs_buses = [0, 9, 17, 24, 32]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(1 + len(self.obs_buses),), dtype=np.float32)

        # Thời gian
        self.t = 0
        self.T = 24

        # Profile tải ngày–đêm
        self.alpha_profile = [0.6,0.6,0.7,0.8,0.9,1.0,1.1,1.2,
                              1.3,1.35,1.3,1.15,1.05,1.0,1.05,1.2,
                              1.35,1.4,1.3,1.1,0.95,0.85,0.75,0.65]

        # Profile PV/Wind (giả định đơn giản)
        self.pv_profile   = [0,0,0,0,0.1,0.3,0.5,0.8,1.0,1.0,0.9,0.7,
                             0.5,0.3,0.2,0.1,0,0,0,0,0,0,0,0]  # MW
        self.wind_profile = [0.3,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                             0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.3,0.3,0.3,0.3,0.3]

        # Lưu tải gốc
        self.base_p = self.net.load.p_mw.values.copy()
        self.base_q = self.net.load.q_mvar.values.copy()

        # Trọng số reward
        self.w_loss = 1.0
        self.w_v = 8.0
        self.w_soc = 2.0
        self.w_ramp = 0.2
        self.w_res = 0.5   # trọng số thưởng hấp thụ RES

        # Theo dõi ramp
        self.prev_p_bess = 0.0
        self.ramp_max = 0.2 * self.max_power

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc = 0.5
        self.t = 0
        self.prev_p_bess = 0.0
        self.net.load.p_mw = self.base_p
        self.net.load.q_mvar = self.base_q
        bess_idx = self.net.sgen[self.net.sgen.name == "BESS"].index
        if len(bess_idx) > 0:
            self.net.sgen.drop(bess_idx, inplace=True)
        pp.runpp(self.net, numba=False)
        return self._get_obs(), {}

    def step(self, action):
        act = float(action[0])
        p_bess = act * self.max_power
        self.soc = np.clip(self.soc + act * self.soc_step, 0.0, 1.0)

        # Cập nhật tải
        factor = self.alpha_profile[self.t]
        self.net.load.p_mw = self.base_p * factor
        self.net.load.q_mvar = self.base_q * factor

        # Cập nhật BESS
        bess_idx = self.net.sgen[self.net.sgen.name == "BESS"].index
        if len(bess_idx) > 0:
            self.net.sgen.drop(bess_idx, inplace=True)
        pp.create_sgen(self.net, bus=17, p_mw=p_bess, q_mvar=0.0, name="BESS")

        # Chạy power flow
        pp.runpp(self.net, numba=False)
        loss = float(self.net.res_line.pl_mw.sum())
        voltages = self.net.res_bus.vm_pu.values

        # Penalties
        penalty_v = float(np.sum(np.maximum(0, np.abs(voltages - 1.0) - 0.05)))
        penalty_soc = 0.0 if (self.soc_low <= self.soc <= self.soc_high) else 1.0
        ramp = abs(p_bess - self.prev_p_bess)
        penalty_ramp = max(0.0, ramp - self.ramp_max)

        # Thưởng hấp thụ RES
        p_res = self.pv_profile[self.t] + self.wind_profile[self.t]
        absorbed = min(-p_bess, p_res) if p_bess < 0 else 0.0
        reward_res = self.w_res * absorbed

        reward = -(self.w_loss * loss) - (self.w_v * penalty_v) - (self.w_soc * penalty_soc) - (self.w_ramp * penalty_ramp) + reward_res

        obs = self._get_obs()
        self.t += 1
        terminated = (self.t >= self.T)
        truncated = False
        info = {
            "loss": loss,
            "soc": float(self.soc),
            "BESS_P_MW": float(p_bess),
            "Reward_RES": reward_res,
            "PV": self.pv_profile[self.t-1],
            "Wind": self.wind_profile[self.t-1]
        }
        self.prev_p_bess = p_bess
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        if self.net.res_bus.empty:
            pp.runpp(self.net, numba=False)
        voltages = self.net.res_bus.vm_pu.values
        obs = [self.soc] + [float(voltages[i]) for i in self.obs_buses]
        return np.array(obs, dtype=np.float32)

# -----------------------------
# 2) Huấn luyện PPO dài
# -----------------------------
env = DummyVecEnv([lambda: IEEE33EnvConvergedRES()])
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=1024, batch_size=1024, gamma=0.99)
model.learn(total_timesteps=200_000)

# -----------------------------
# 3) Đánh giá 24h
# -----------------------------
obs = env.reset()
rows = []
for h in range(24):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    rows.append({
        "Hour": h,
        "SoC": infos[0]["soc"],
        "Reward": rewards[0],
        "Loss": infos[0]["loss"],
        "BESS_P_MW": infos[0]["BESS_P_MW"],
        "Reward_RES": infos[0]["Reward_RES"],
        "PV": infos[0]["PV"],
        "Wind": infos[0]["Wind"]
    })
    if dones[0]:
        break

df = pd.DataFrame(rows)
df.to_csv("scenario3_converged_res.csv", index=False, encoding="utf-8-sig")

# -----------------------------
# 4) Vẽ biểu đồ
# -----------------------------
fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

axs[0].plot(df["Hour"], df["SoC"], marker="o", label="SoC")
axs[0].axhline(0.2, color="gray", linestyle="--")
axs[0].axhline(0.9, color="gray", linestyle="--")
axs[0].set_ylabel("SoC")
axs[0].legend()

axs[1].plot(df["Hour"], df["BESS_P_MW"], marker="s", color="purple", label="BESS P (MW)")
axs[1].set_ylabel("BESS P (MW)")
axs[1].legend()

axs[2]