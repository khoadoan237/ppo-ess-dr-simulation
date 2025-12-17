# run_one.py
import os

# Cấu hình mô hình và seed
config_path = "configs/ppoess.yaml"
seed = 42

# Lệnh chạy mô phỏng
os.system(f"python main.py --config {config_path} --seed {seed}")