# ============================================
# KỊCH BẢN 1: BASELINE (KHÔNG ESS, KHÔNG DR)
# Tối ưu hóa vận hành Microgrid IEEE 33-bus
# Bám sát ràng buộc Chương 3 + Chương 4
# ============================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandapower as pp
import pandapower.networks as nw

# ====================================================
# 0. CÀI ĐẶT SEED CHO TOÀN BỘ QUÁ TRÌNH
# ====================================================
GLOBAL_SEED = 1
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


# ====================================================
# 1. KỊCH BẢN BASELINE – KHÔNG ESS, KHÔNG DR
# ====================================================
class BaselineIEEE33:
    def __init__(self, seed=1):
        self.seed = seed
        np.random.seed(seed)

        # Load IEEE 33-bus
        self.net = nw.case33bw()

        # Time horizon
        self.T = 24
        self.t = 0

        # Profiles
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

        self.grid_price = np.full(self.T, 0.1, dtype=float)

        # Backup load
        self.base_p = self.net.load.p_mw.values.copy()
        self.base_q = self.net.load.q_mvar.values.copy()

        # Create PV + Wind generators
        self._init_sgen()

    def _init_sgen(self):
        if len(self.net.sgen):
            self.net.sgen.drop(self.net.sgen.index, inplace=True)

        # PV tại bus 18
        pp.create_sgen(self.net, bus=18, p_mw=0.0, q_mvar=0.0, name="PV")

        # Wind tại bus 25
        pp.create_sgen(self.net, bus=25, p_mw=0.0, q_mvar=0.0, name="Wind")

    def run_24h(self):
        rows = []

        for t in range(self.T):
            alpha = self.alpha_profile[t]
            pv = self.pv_profile[t]
            wind = self.wind_profile[t]

            # Load update
            self.net.load.p_mw = self.base_p * alpha
            self.net.load.q_mvar = self.base_q * alpha

            # PV + Wind update
            self.net.sgen.loc[self.net.sgen.name == "PV", "p_mw"] = pv
            self.net.sgen.loc[self.net.sgen.name == "Wind", "p_mw"] = wind

            # Power flow
            pp.runpp(self.net, numba=False)

            # Metrics
            loss = float(self.net.res_line.pl_mw.sum())
            voltages = self.net.res_bus.vm_pu.values

            # Grid import only
            p_grid = float(self.net.res_ext_grid.p_mw.sum())
            p_grid_import = max(0.0, p_grid)
            c_grid = p_grid_import * self.grid_price[t]

            # PV curtail = 0 trong baseline
            pv_curtail = 0.0

            # Voltage deviation
            v_dev = float(np.max(np.abs(voltages - 1.0)))

            rows.append({
                "Hour": t,
                "LoadFactor": alpha,
                "PV": pv,
                "Wind": wind,
                "GridCost": c_grid,
                "Loss": loss,
                "VoltageDev": v_dev,
                "PV_Curtail": pv_curtail
            })

        return pd.DataFrame(rows)


# ====================================================
# 2. CHẠY KỊCH BẢN 24H & XUẤT CSV
# ====================================================
if __name__ == "__main__":
    sim = BaselineIEEE33(seed=GLOBAL_SEED)
    df = sim.run_24h()

    df.to_csv("scenario1_baseline.csv", index=False, encoding="utf-8-sig")
    print("Saved results to scenario1_baseline.csv")

    # ====================================================
    # 3. VẼ BIỂU ĐỒ
    # ====================================================
    fig, axs = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    # (0) Load factor
    axs[0].plot(df["Hour"], df["LoadFactor"], marker="o", label="Load Factor")
    axs[0].set_ylabel("Load Factor")
    axs[0].legend(); axs[0].grid(True)

    # (1) PV + Wind
    axs[1].plot(df["Hour"], df["PV"], marker="s", label="PV (MW)")
    axs[1].plot(df["Hour"], df["Wind"], marker="^", label="Wind (MW)")
    axs[1].set_ylabel("Power (MW)")
    axs[1].legend(); axs[1].grid(True)

    # (2) Grid Cost
    axs[2].plot(df["Hour"], df["GridCost"], marker="x", color="red", label="Grid Cost")
    axs[2].set_ylabel("Grid Cost")
    axs[2].legend(); axs[2].grid(True)

    # (3) Loss
    axs[3].plot(df["Hour"], df["Loss"], marker="d", color="purple", label="Loss (MW)")
    axs[3].set_ylabel("Loss (MW)")
    axs[3].legend(); axs[3].grid(True)

    # (4) Voltage deviation
    axs[4].plot(df["Hour"], df["VoltageDev"], marker="o", color="blue", label="Voltage Dev")
    axs[4].set_xlabel("Hour")
    axs[4].set_ylabel("Voltage Dev (pu)")
    axs[4].legend(); axs[4].grid(True)

    plt.tight_layout()
    plt.show()