# ============================================
# KỊCH BẢN 2: PPO + ESS (PPO-ESS, KHÔNG DR)
# Tối ưu hóa vận hành Microgrid IEEE 33-bus
# Bám sát ràng buộc Chương 3 + Chương 4
# ============================================

# pip install gymnasium stable-baselines3 pandapower pandas matplotlib numpy torch

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import pandapower as pp
import pandapower.networks as nw

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


# ====================================================
# 0. CÀI ĐẶT SEED CHO TOÀN BỘ QUÁ TRÌNH
# ====================================================
GLOBAL_SEED = 1
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)


# ====================================================
# 1. MÔI TRƯỜNG DRL CHO KỊCH BẢN PPO + ESS (KHÔNG DR)
# ====================================================
class IEEE33EnvPPOESS(gym.Env):
    """
    Môi trường DRL cho Microgrid IEEE 33-bus:
    - Nguồn: PV, gió (wind) theo profile 24h
    - ESS với ràng buộc SoC, công suất, hiệu suất
    - KHÔNG có DR: phụ tải chỉ theo alpha_profile
    - Ràng buộc điện áp: 0.95–1.05 pu
    - Reward: chi phí lưới, PV curtail, điện áp, ESS, ramping
    """

    metadata = {"render.modes": []}

    def __init__(self, seed: int = 1):
        super().__init__()

        # ----- seed nội bộ môi trường -----
        self._np_random, _ = gym.utils.seeding.np_random(seed)

        # ----- Lưới IEEE 33-bus -----
        self.net = nw.case33bw()

        # ----- Cấu hình ESS -----
        self.soc = 0.5
        self.soc_min = 0.2
        self.soc_max = 0.9
        self.p_ess_max = 0.2      # MW (P_charge_max = P_discharge_max)
        self.eta_ch = 0.95        # hiệu suất sạc
        self.eta_dis = 0.95       # hiệu suất xả
        self.prev_p_ess = 0.0

        # ----- Action space -----
        # [a_ESS, a_PV_curtail]
        #  a_ESS      ∈ [-1, 1]   => P_ESS ∈ [-P_max, +P_max]
        #  a_PV_cur   ∈ [ 0, 1]   => PV_curtail_ratio
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # ----- Observation space -----
        # [SoC, PV_profile_t, Load_factor_t, Grid_price_t, VoltageDev_t]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.5, 1.5, 1.0, 1.0], dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )

        # ----- Thời gian mô phỏng -----
        self.t = 0
        self.T = 24  # 24 bước = 24h

        # ----- Profiles 24h: tải, PV, gió -----
        self.alpha_profile = np.array([
            0.6,0.6,0.7,0.8,0.9,1.0,1.1,1.2,
            1.3,1.35,1.3,1.15,1.05,1.0,1.05,1.2,
            1.35,1.4,1.3,1.1,0.95,0.85,0.75,0.65
        ], dtype=float)

        self.pv_profile = np.array([
            0,0,0,0,0.1,0.3,0.5,0.8,1.0,1.0,0.9,0.7,
            0.5,0.3,0.2,0.1,0,0,0,0,0,0,0,0
        ], dtype=float)

        self.wind_profile = np.array([
            0.3,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,
            0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.3,0.3,0.3,0.3,0.3
        ], dtype=float)

        # Giá điện (có thể nâng cấp thành TOU)
        self.grid_price = np.full(self.T, 0.1, dtype=float)

        # Backup tải gốc
        self.base_p = self.net.load.p_mw.values.copy()
        self.base_q = self.net.load.q_mvar.values.copy()

        # ----- Tham số reward -----
        self.w_grid_cost   = 1.0   # chi phí lưới
        self.w_pv_curtail  = 0.3   # mất mát PV
        self.w_v_dev       = 10.0  # độ lệch điện áp
        self.w_soc_safe    = 2.0   # vi phạm dải SoC
        self.w_ess_deg     = 0.5   # hao mòn ESS (ΔSoC)
        self.w_ramp        = 0.2   # ramping ESS
        self.ramp_max      = 0.2 * self.p_ess_max

        # Tạo sgen PV / Wind / ESS
        self._init_sgen()

    # ------------------------------
    # Khởi tạo các nguồn sgen
    # ------------------------------
    def _init_sgen(self):
        if len(self.net.sgen):
            self.net.sgen.drop(self.net.sgen.index, inplace=True)
        pp.create_sgen(self.net, bus=17, p_mw=0.0, q_mvar=0.0, name="ESS")
        pp.create_sgen(self.net, bus=18, p_mw=0.0, q_mvar=0.0, name="PV")
        pp.create_sgen(self.net, bus=25, p_mw=0.0, q_mvar=0.0, name="Wind")

    # ------------------------------
    # Reset môi trường
    # ------------------------------
    def reset(self, seed=None, options=None):
        # Seed lại rng nếu cần
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        self.soc = 0.5
        self.t = 0
        self.prev_p_ess = 0.0

        self.net.load.p_mw = self.base_p
        self.net.load.q_mvar = self.base_q
        self._init_sgen()
        pp.runpp(self.net, numba=False)

        return self._get_obs(), {}

    # ------------------------------
    # Một bước mô phỏng
    # ------------------------------
    def step(self, action):
        a_ess, a_pv_curtail = np.array(action, dtype=np.float32)

        # Clip action
        a_ess = float(np.clip(a_ess, -1.0, 1.0))
        a_pv_curtail = float(np.clip(a_pv_curtail, 0.0, 1.0))

        # Công suất ESS (MW) >0: xả, <0: sạc
        p_ess = a_ess * self.p_ess_max

        # Giới hạn SoC: không cho xả nếu SoC thấp, không cho sạc nếu SoC cao
        if self.soc <= self.soc_min and p_ess > 0:
            p_ess = 0.0
        if self.soc >= self.soc_max and p_ess < 0:
            p_ess = 0.0

        # Cập nhật SoC với hiệu suất (Δt = 1h, dung lượng chuẩn hóa)
        if p_ess >= 0:  # discharge -> SoC giảm
            delta_soc = - p_ess / (self.p_ess_max * self.eta_dis)
        else:           # charge -> SoC tăng (p_ess < 0)
            delta_soc = - p_ess * self.eta_ch / self.p_ess_max

        self.soc = float(np.clip(self.soc + delta_soc, 0.0, 1.0))

        # Phụ tải KHÔNG DR: chỉ theo alpha_profile
        alpha_t = float(self.alpha_profile[self.t])
        base_p_t = self.base_p * alpha_t
        base_q_t = self.base_q * alpha_t

        self.net.load.p_mw = base_p_t
        self.net.load.q_mvar = base_q_t

        # PV và gió
        pv_raw = float(self.pv_profile[self.t])
        pv_eff = pv_raw * (1.0 - a_pv_curtail)
        wind_eff = float(self.wind_profile[self.t])

        self.net.sgen.loc[self.net.sgen.name == "ESS",  "p_mw"] = p_ess
        self.net.sgen.loc[self.net.sgen.name == "PV",   "p_mw"] = pv_eff
        self.net.sgen.loc[self.net.sgen.name == "Wind", "p_mw"] = wind_eff

        # Chạy trào lưu công suất
        pp.runpp(self.net, numba=False)

        # Tính các chỉ tiêu
        loss = float(self.net.res_line.pl_mw.sum())
        voltages = self.net.res_bus.vm_pu.values

        # Chỉ cho phép mua điện (không bán)
        p_grid = float(self.net.res_ext_grid.p_mw.sum()) if len(self.net.res_ext_grid) else 0.0
        p_grid_import = max(0.0, p_grid)
        c_grid = p_grid_import * float(self.grid_price[self.t])

        # PV curtail
        p_curtail = max(0.0, pv_raw - pv_eff)

        # Voltage deviation: max |V-1|
        v_dev = float(np.max(np.abs(voltages - 1.0)))

        # SoC ra khỏi dải an toàn
        soc_violation = float(
            max(0.0, self.soc_min - self.soc) + max(0.0, self.soc - self.soc_max)
        )

        # Ramping của ESS
        ramp = abs(p_ess - self.prev_p_ess)
        ramp_violation = max(0.0, ramp - self.ramp_max)

        # Proxy cho hao mòn ESS (tổng |ΔSoC|)
        ess_deg_proxy = abs(delta_soc)

        # Reward đa mục tiêu (dấu trừ vì ta muốn minimize các đại lượng này)
        reward = -(
            self.w_grid_cost  * c_grid +
            self.w_pv_curtail * p_curtail +
            self.w_v_dev      * v_dev +
            self.w_soc_safe   * soc_violation +
            self.w_ess_deg    * ess_deg_proxy
        ) - self.w_ramp * ramp_violation

        # Lưu info cho logging
        info = {
            "loss": loss,
            "soc": float(self.soc),
            "P_ESS": float(p_ess),
            "PV": float(pv_eff),
            "PV_raw": float(pv_raw),
            "Wind": float(wind_eff),
            "GridCost": float(c_grid),
            "PV_Curtail": float(p_curtail),
            "VoltageDev": float(v_dev),
            "SoC_Violation": float(soc_violation),
            "ESS_DegProxy": float(ess_deg_proxy),
        }

        self.prev_p_ess = p_ess
        obs = self._get_obs()

        self.t += 1
        terminated = (self.t >= self.T)
        truncated = False

        return obs, reward, terminated, truncated, info

    # ------------------------------
    # Tạo vector quan sát
    # ------------------------------
    def _get_obs(self):
        if self.net.res_bus.empty:
            pp.runpp(self.net, numba=False)
        voltages = self.net.res_bus.vm_pu.values
        v_dev = float(np.max(np.abs(voltages - 1.0)))
        obs = np.array([
            float(self.soc),
            float(self.pv_profile[self.t]),
            float(self.alpha_profile[self.t]),
            float(self.grid_price[self.t]),
            v_dev
        ], dtype=np.float32)
        return obs


# ====================================================
# 2. HUẤN LUYỆN PPO CHO KỊCH BẢN PPO-ESS
# ====================================================
if __name__ == "__main__":
    seed = GLOBAL_SEED

    # Tạo env vector hóa – KHÔNG gọi reset(seed=...) trên DummyVecEnv
    env = DummyVecEnv([lambda: IEEE33EnvPPOESS(seed=seed)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=1024,
        gamma=0.99,
        seed=seed
    )

    model.learn(total_timesteps=2000)

    # ====================================================
    # 3. ĐÁNH GIÁ 24H & XUẤT CSV
    # ====================================================
    obs = env.reset()
    rows = []

    for h in range(24):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        info = infos[0]
        rows.append({
            "Hour": h,
            "SoC": info["soc"],
            "Reward": float(rewards[0]),
            "Loss": info["loss"],
            "P_ESS": info["P_ESS"],
            "PV": info["PV"],
            "PV_raw": info["PV_raw"],
            "Wind": info["Wind"],
            "GridCost": info["GridCost"],
            "PV_Curtail": info["PV_Curtail"],
            "VoltageDev": info["VoltageDev"],
            "SoC_Violation": info["SoC_Violation"],
            "ESS_DegProxy": info["ESS_DegProxy"],
        })

        if dones[0]:
            break

    df = pd.DataFrame(rows)
    df.to_csv("scenario2_ppo_ess.csv", index=False, encoding="utf-8-sig")
    print("Saved results to scenario2_ppo_ess.csv")

    # ====================================================
    # 4. VẼ MỘT SỐ BIỂU ĐỒ CƠ BẢN
    # ====================================================
    fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

    # (0) SoC
    axs[0].plot(df["Hour"], df["SoC"], marker="o", label="SoC")
    axs[0].axhline(0.2, color="gray", linestyle="--", linewidth=1)
    axs[0].axhline(0.9, color="gray", linestyle="--", linewidth=1)
    axs[0].set_ylabel("SoC")
    axs[0].legend(); axs[0].grid(True)

    # (1) P_ESS
    axs[1].plot(df["Hour"], df["P_ESS"], marker="s", color="purple", label="P_ESS (MW)")
    axs[1].set_ylabel("P_ESS (MW)")
    axs[1].legend(); axs[1].grid(True)

    # (2) Grid cost
    axs[2].plot(df["Hour"], df["GridCost"], marker="^", color="red", label="Grid Cost")
    axs[2].set_ylabel("Grid Cost")
    axs[2].legend(); axs[2].grid(True)

    # (3) PV utilization
    pv_total_raw = (df["PV_raw"]).sum()
    pv_used = df["PV"].sum()
    pv_util = 100.0 * pv_used / (pv_total_raw + 1e-9)
    axs[3].bar(["PV Util (%)"], [pv_util], color="green")
    axs[3].set_ylabel("%"); axs[3].grid(True)

    # (4) ESS cycles (xấp xỉ bằng số lần đổi chiều dSoC)
    soc = df["SoC"].values
    diff = np.diff(soc)
    sign = np.sign(diff)
    nonzero = sign[sign != 0]
    cycles_soc = int(np.sum(nonzero[1:] * nonzero[:-1] == -1))
    axs[4].bar(["ESS Cycles (SoC)"], [cycles_soc], color="purple")
    axs[4].set_ylabel("Count"); axs[4].grid(True)
    axs[4].text(0, cycles_soc + 0.05, str(cycles_soc), ha="center")

    # (5) Voltage deviation
    axs[5].plot(df["Hour"], df["VoltageDev"], marker="x", color="blue", label="Voltage Dev (pu)")
    axs[5].set_xlabel("Hour")
    axs[5].set_ylabel("Voltage Dev")
    axs[5].legend(); axs[5].grid(True)

    plt.tight_layout()
    plt.show()