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
# 1) Môi trường islanded mô phỏng (giữ slack ảo, phạt import)
# -----------------------------
class IEEE33EnvIslandedVirtualSlack(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.net = nw.case33bw()  # có sẵn ext_grid tại bus 0 (slack ảo)

        # Tham số BESS
        self.soc = 0.5
        self.max_power = 0.25    # MW: biên sạc/xả
        self.soc_step = 0.05
        self.soc_low, self.soc_high = 0.2, 0.9

        # Quan sát và hành động
        self.obs_buses = [0, 9, 17, 24, 32]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=2.0, shape=(1 + len(self.obs_buses),), dtype=np.float32)

        # Thời gian
        self.t = 0
        self.T = 24

        # Profile tải ngày–đêm
        self.alpha_profile = [0.6,0.7,0.8,1.0,1.2,1.3,1.4,1.3,1.2,1.0,0.8,0.7,
                              0.6,0.7,0.9,1.1,1.3,1.4,1.2,1.0,0.8,0.7,0.6,0.6]

        # Profile PV/Wind (MW)
        self.pv_profile   = [0,0,0,0,0.2,0.5,0.8,1.0,1.0,0.8,0.5,0.2,
                             0,0,0,0,0,0,0,0,0,0,0,0]
        self.wind_profile = [0.3]*24

        # Lưu tải gốc
        self.base_p = self.net.load.p_mw.values.copy()
        self.base_q = self.net.load.q_mvar.values.copy()

        # Trọng số reward
        self.w_v = 10.0         # phạt vi phạm điện áp
        self.w_soc = 3.0        # phạt SoC vượt ngưỡng
        self.w_unserved = 25.0  # phạt thiếu công suất (ưu tiên đảm bảo tải)
        self.w_res = 0.7        # thưởng hấp thụ RES khi sạc
        self.w_grid = 50.0      # phạt import từ lưới (slack ảo): càng lớn càng “islanded”

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.soc = 0.5
        self.t = 0
        # reset tải
        self.net.load.p_mw = self.base_p
        self.net.load.q_mvar = self.base_q
        # xóa các sgen cũ
        if len(self.net.sgen.index) > 0:
            self.net.sgen.drop(self.net.sgen.index, inplace=True)
        # giữ ext_grid tại bus 0 làm slack ảo, đặt vm_pu = 1.0
        if len(self.net.ext_grid.index) == 0:
            pp.create_ext_grid(self.net, bus=0, vm_pu=1.0, name="VirtualSlack")
        else:
            self.net.ext_grid.vm_pu = 1.0

        # chạy nhẹ để init res
        try:
            pp.runpp(self.net, numba=False)
        except Exception:
            pass
        return self._get_obs(), {}

    def step(self, action):
        act = float(np.clip(action[0], -1.0, 1.0))
        p_bess = act * self.max_power
        self.soc = float(np.clip(self.soc + act*self.soc_step, 0.0, 1.0))

        # cập nhật tải
        factor = float(self.alpha_profile[self.t])
        self.net.load.p_mw = self.base_p * factor
        self.net.load.q_mvar = self.base_q * factor

        # xóa sgen cũ
        if len(self.net.sgen.index) > 0:
            self.net.sgen.drop(self.net.sgen.index, inplace=True)

        # thêm RES (bus 0) và BESS (bus 17)
        pp.create_sgen(self.net, bus=0, p_mw=float(self.pv_profile[self.t] + self.wind_profile[self.t]),
                       q_mvar=0.0, name="RES")
        pp.create_sgen(self.net, bus=17, p_mw=float(p_bess), q_mvar=0.0, name="BESS")

        # chạy power flow an toàn
        converged = True
        try:
            pp.runpp(self.net, numba=False)
            voltages = self.net.res_bus.vm_pu.values
            loss = float(self.net.res_line.pl_mw.sum())
            # công suất từ lưới (ext_grid). p_mw > 0: lưới cấp vào, p_mw < 0: phát ra lưới
            p_grid = float(self.net.res_ext_grid.p_mw.sum()) if len(self.net.res_ext_grid.index) > 0 else 0.0
        except Exception:
            converged = False
            voltages = np.ones(len(self.net.bus))
            loss = 0.0
            p_grid = 0.0

        # penalties
        penalty_v = float(np.sum(np.maximum(0.0, np.abs(voltages - 1.0) - 0.05)))
        penalty_soc = 0.0 if (self.soc_low <= self.soc <= self.soc_high) else 1.0

        # thiếu công suất (không chia 0)
        total_gen = float(self.pv_profile[self.t] + self.wind_profile[self.t] + max(p_bess, 0.0))
        total_load = float(np.sum(self.net.load.p_mw.values))
        unserved = float(max(0.0, total_load - total_gen))

        # thưởng hấp thụ RES
        absorbed = float(min(-p_bess, self.pv_profile[self.t] + self.wind_profile[self.t])) if p_bess < 0.0 else 0.0
        reward_res = self.w_res * absorbed

        # phạt import từ lưới (muốn islanded → p_grid_import gần 0)
        p_grid_import = max(0.0, p_grid)  # chỉ phạt chiều lưới cấp vào
        penalty_grid = self.w_grid * p_grid_import

        reward = -(self.w_v * penalty_v) \
                 -(self.w_soc * penalty_soc) \
                 -(self.w_unserved * unserved) \
                 -(penalty_grid) \
                 + (reward_res)

        # đảm bảo obs hữu hạn
        obs = self._get_obs()
        self.t += 1
        terminated = bool(self.t >= self.T)
        truncated = False

        info = {
            "soc": self.soc,
            "loss": loss,
            "unserved": unserved,
            "BESS_P_MW": p_bess,
            "PV": float(self.pv_profile[self.t-1]),
            "Wind": float(self.wind_profile[self.t-1]),
            "P_grid": p_grid,
            "ConvergedPF": converged
        }
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        if self.net.res_bus.empty:
            voltages = np.ones(len(self.net.bus))
        else:
            voltages = self.net.res_bus.vm_pu.values
        obs = [float(self.soc)] + [float(voltages[i]) for i in self.obs_buses]
        return np.clip(np.array(obs, dtype=np.float32), self.observation_space.low, self.observation_space.high)

# -----------------------------
# 2) Huấn luyện PPO (dùng SDE để tránh scale=0)
# -----------------------------
env = DummyVecEnv([lambda: IEEE33EnvIslandedVirtualSlack()])

policy_kwargs = dict(
    net_arch=[64, 64],
    log_std_init=-2.0,   # std ~ exp(-2) ~ 0.135
    ortho_init=True
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=1024,
    gamma=0.99,
    clip_range=0.2,
    use_sde=True,
    sde_sample_freq=4,
    policy_kwargs=policy_kwargs
)

model.learn(total_timesteps=5000)

# -----------------------------
# 3) Đánh giá 24h và ghi CSV
# -----------------------------
obs = env.reset()
rows = []
for h in range(24):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    info = infos[0]
    rows.append({
        "Hour": h,
        "SoC": float(info["soc"]),
        "Reward": float(rewards[0]),
        "Loss": float(info["loss"]),
        "Unserved": float(info["unserved"]),
        "BESS_P_MW": float(info["BESS_P_MW"]),
        "PV": float(info["PV"]),
        "Wind": float(info["Wind"]),
        "P_grid": float(info["P_grid"]),
        "ConvergedPF": bool(info["ConvergedPF"])
    })
    if dones[0]:
        break

df = pd.DataFrame(rows)
df.to_csv("scenario4_islanded_virtual_slack.csv", index=False, encoding="utf-8-sig")
print("✅ Đã lưu scenario4_islanded_virtual_slack.csv")

# -----------------------------
# 4) Vẽ biểu đồ
# -----------------------------
fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

axs[0].plot(df["Hour"], df["SoC"], marker="o", label="SoC")
axs[0].axhline(0.2, color="gray", linestyle="--"); axs[0].axhline(0.9, color="gray", linestyle="--")
axs[0].set_ylabel("SoC"); axs[0].legend(); axs[0].grid(True, linestyle="--", alpha=0.4)

axs[1].plot(df["Hour"], df["BESS_P_MW"], marker="s", color="purple", label="BESS P (MW)")
axs[1].set_ylabel("BESS P (MW)"); axs[1].legend(); axs[1].grid(True, linestyle="--", alpha=0.4)

axs[2].plot(df["Hour"], df["PV"], marker="^", color="orange", label="PV")
axs[2].plot(df["Hour"], df["Wind"], marker="v", color="green", label="Wind")
axs[2].set_ylabel("RES (MW)"); axs[2].legend(); axs[2].grid(True, linestyle="--", alpha=0.4)

axs[3].plot(df["Hour"], df["Unserved"], marker="D", color="red", label="Unserved (MW)")
axs[3].set_ylabel("Unserved (MW)"); axs[3].legend(); axs[3].grid(True, linestyle="--", alpha=0.4)

axs[4].plot(df["Hour"], df["P_grid"], marker="o", color="brown", label="Grid import (MW)")
axs[4].set_ylabel("P_grid (MW)"); axs[4].legend(); axs[4].grid(True, linestyle="--", alpha=0.4)

axs[5].plot(df["Hour"], df["Reward"], marker="o", color="blue", label="Reward")
axs[5].set_ylabel("Reward"); axs[5].set_xlabel("Hour"); axs[5].legend(); axs[5].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("scenario4_islanded_virtual_slack_plots.png", dpi=150)
plt.show()
print("📈 Đã vẽ biểu đồ: scenario4_islanded_virtual_slack_plots.png")