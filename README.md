
README – PPO‑ESS‑DR Simulation Framework
Dự án mô phỏng vận hành lưới phân phối có tích hợp hệ thống lưu trữ năng lượng (ESS) và điều chỉnh phụ tải (Demand Response – DR), được điều khiển bằng thuật toán học tăng cường Proximal Policy Optimization (PPO).
1. 	Yêu cầu hệ thống
• 	Python 3.8 – 3.11
• 	pip ≥ 21.0
• 	Windows / Linux / macOS
• 	Khuyến nghị sử dụng môi trường ảo (venv)
2. 	Cài đặt môi trường
Tạo môi trường ảo (Windows):
python -m venv venv
venv\Scripts\activate
Tạo môi trường ảo (Linux/macOS):
python3 -m venv venv
source venv/bin/activate
Cài đặt thư viện:
pip install -r requirements.txt
Nếu chưa có file requirements.txt:
pip freeze > requirements.txt
3. 	Cấu trúc thư mục
configs/ – chứa file cấu hình
envs/ – môi trường mô phỏng
models/ – lưu mô hình PPO
results/ – kết quả mô phỏng
scripts/ – script chạy mô phỏng
main.py – chương trình chính
requirements.txt – danh sách thư viện
4. 	Cách chạy mô phỏng
Chạy một lần với seed cố định:
python main.py --config configs/ppoess.yaml --seed 42
Chạy nhiều lần để kiểm định thống kê:
python scripts/batch_run.py
Danh sách seed có thể chỉnh trong scripts/batch_run.py.
5. 	File cấu hình mô phỏng
baseline.yaml – không ESS, không DR
train_steps: 50000
save_path: models/baseline_model
ess_capacity: 0
ess_power: 0
dr_ratio: 0.0
ppoess.yaml – có ESS
train_steps: 50000
save_path: models/ppoess_model
ess_capacity: 5000
ess_power: 2000
dr_ratio: 0.0
ppoessdr.yaml – có ESS + DR
train_steps: 50000
save_path: models/ppoessdr_model
ess_capacity: 5000
ess_power: 2000
dr_ratio: 0.2
6. 	Kết quả mô phỏng
Kết quả được lưu trong thư mục results/, bao gồm:
• 	Reward trung bình
• 	Chi phí lưới (GridCost)
• 	Công suất ESS (P_ESS)
• 	Trạng thái sạc ESS (SoC)
• 	Độ lệch điện áp (Voltage deviation)
• 	Mức sử dụng PV (PV_util)
7. 	Lưu ý khi đưa lên GitHub
Tạo file .gitignore để loại trừ file nặng:
venv/
pycache/
*.pyc
*.pyd
*.dll
*.log
*.csv
*.xlsx
Nếu lỡ commit venv rồi, gỡ ra:
git rm -r --cached venv
8. 	Hướng dẫn đưa dự án lên GitHub
Khởi tạo Git:
git init
Thêm remote GitHub:
git remote add origin https://github.com/khoadoan237/ppo-ess-dr-simulation.git
Commit và đẩy mã nguồn:
git add .
git commit -m "Initial commit with PPO-ESS-DR simulation"
git push --set-upstream origin master
9. 	Thông tin tác giả
   Đoàn Trịnh Đăng Khoa
Email: doantrinhdangkhoa2001@gmail.com
GitHub: https://github.com/khoadoan237
