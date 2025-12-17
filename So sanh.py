import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df1 = pd.read_csv("scenario1_baseline.csv")
df2 = pd.read_csv("scenario2_warmup.csv")
df4 = pd.read_csv("scenario4_islanded_virtual_slack.csv")

# Tạo figure 2x2
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Kịch bản 1 - Baseline
axs[0,0].plot(df1["Hour"], df1["Reward"], label="Reward", color="blue")
if "Loss" in df1.columns:
    axs[0,0].plot(df1["Hour"], df1["Loss"], label="Loss", color="red")
axs[0,0].set_title("Scenario 1 - Baseline")
axs[0,0].legend(); axs[0,0].grid(True, linestyle="--", alpha=0.4)

# Kịch bản 2 - Warm-up
axs[0,1].plot(df2["Hour"], df2["Reward"], label="Reward", color="blue")
if "SoC" in df2.columns:
    axs[0,1].plot(df2["Hour"], df2["SoC"], label="SoC", color="green")
axs[0,1].set_title("Scenario 2 - Warm-up")
axs[0,1].legend(); axs[0,1].grid(True, linestyle="--", alpha=0.4)

# Kịch bản 4 - Islanded
axs[1,1].plot(df4["Hour"], df4["Reward"], label="Reward", color="blue")
if "SoC" in df4.columns:
    axs[1,1].plot(df4["Hour"], df4["SoC"], label="SoC", color="green")
if "Unserved" in df4.columns:
    axs[1,1].plot(df4["Hour"], df4["Unserved"], label="Unserved", color="red")
axs[1,1].set_title("Scenario 4 - Islanded")
axs[1,1].legend(); axs[1,1].grid(True, linestyle="--", alpha=0.4)

# Nhãn chung
for ax in axs.flat:
    ax.set_xlabel("Hour")
    ax.set_ylabel("Value")

plt.tight_layout()
plt.savefig("compare_all_scenarios_combined.png", dpi=150)
plt.show()