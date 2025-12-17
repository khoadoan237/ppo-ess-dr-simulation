import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandapower as pp
import pandapower.networks as nw
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Môi trường Baseline (nối lưới, có RES, không BESS)
# -----------------------------
class IEEE33BaselineEnvRES(gym.Env):
    def __init__(self):
        super().__init__()
        self.net = nw.case33bw()
        self.t = 0
        self.obs_buses = [0, 9, 17, 24, 32]

        # Không có hành động (BESS off)
        self.action_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=2.0,
                                            shape=(len(self.obs_buses),), dtype=np.float32)

        # Profile tải ngày–đêm
        self.alpha_profile = [0.6,0.6,0.7,0.8,0.9,1.0,1.1,1.2,
                              1.3,1.35,1.3,1.15,1.05,1.0,1.05,1.2,
                              1.35,1.4,1.3,1.1,0.95,0.85,0.75,0.65]

        # Profile PV/Wind (MW, giả lập)
        self.pv_profile   = [0,0,0,0,0.1,0.3,0.5,0.8,1.0,1.0,0.9,0.7,
                             0.5,0.3,0.2,0.1,0,0,0,0,0,0,0,0]
        self.wind_profile = [0.3,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
                             0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.3,0.3,0.3,0.3,0.3]

        # Lưu tải gốc
        self.base_p = self.net.load.p_mw.values.copy()
        self.base_q = self.net.load.q_mvar.values.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.net.load.p_mw = self.base_p
        self.net.load.q_mvar = self.base_q
        pp.runpp(self.net, numba=False)
        return self._get_obs(), {}

    def step(self, action):
        # Cập nhật tải
        factor = self.alpha_profile[self.t]
        self.net.load.p_mw = self.base_p * factor
        self.net.load.q_mvar = self.base_q * factor

        # Thêm PV/Wind vào bus 0
        res_idx = self.net.sgen[self.net.sgen.name == "RES"].index
        if len(res_idx) > 0:
            self.net.sgen.drop(res_idx, inplace=True)
        pp.create_sgen(self.net, bus=0,
                       p_mw=self.pv_profile[self.t] + self.wind_profile[self.t],
                       q_mvar=0.0, name="RES")

        # Chạy power flow
        pp.runpp(self.net, numba=False)
        voltages = self.net.res_bus.vm_pu.values
        loss = float(self.net.res_line.pl_mw.sum())
        penalty = np.sum(np.maximum(0, np.abs(voltages-1.0)-0.05))
        reward = -loss - penalty

        obs = self._get_obs()
        self.t += 1
        terminated = (self.t >= 24)
        return obs, reward, terminated, False, {"loss": loss, "reward": reward, "voltages": voltages}

    def _get_obs(self):
        if self.net.res_bus.empty:
            pp.runpp(self.net, numba=False)
        voltages = self.net.res_bus.vm_pu.values
        return np.array([voltages[i] for i in self.obs_buses], dtype=np.float32)

# -----------------------------
# 2. Chạy 24h và log kết quả
# -----------------------------
env = IEEE33BaselineEnvRES()
obs, info = env.reset()
rows = []
for h in range(24):
    obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    row = {"Hour": h, "Reward": reward, "Loss": info["loss"]}
    for i, b in enumerate(env.obs_buses):
        row[f"V_bus_{b}"] = info["voltages"][b]
    rows.append(row)
    if terminated:
        break

df = pd.DataFrame(rows)
df.to_csv("scenario1_baseline.csv", index=False, encoding="utf-8-sig")
print("✅ Đã lưu scenario1_baseline.csv")

# -----------------------------
# 3. Vẽ biểu đồ
# -----------------------------
fig = plt.figure(figsize=(12, 8))

# Reward & Loss
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(df["Hour"], df["Reward"], marker="o", color="blue", label="Reward")
ax1.set_ylabel("Reward", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True, linestyle="--", alpha=0.4)

ax1b = ax1.twinx()
ax1b.plot(df["Hour"], df["Loss"], marker="s", color="red", label="Loss (MW)")
ax1b.set_ylabel("Loss (MW)", color="red")
ax1b.tick_params(axis="y", labelcolor="red")

ax1.set_title("Baseline (nối lưới, có RES, không BESS): Reward & Loss theo giờ")
ax1.set_xlabel("Hour")

# Điện áp các bus
ax2 = fig.add_subplot(2, 1, 2)
bus_cols = [c for c in df.columns if c.startswith("V_bus_")]
for c in bus_cols:
    ax2.plot(df["Hour"], df[c], marker=".", label=c)

ax2.axhline(0.95, color="orange", linestyle="--", linewidth=1)
ax2.axhline(1.05, color="orange", linestyle="--", linewidth=1)
ax2.set_ylabel("Voltage (pu)")
ax2.set_xlabel("Hour")
ax2.set_title("Điện áp các bus quan sát theo giờ")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(ncol=2, fontsize=9)

plt.tight_layout()
plt.savefig("scenario1_baseline_plots.png", dpi=150)
plt.show()