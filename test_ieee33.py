import pandapower as pp
import pandapower.networks as nw

# Load lưới IEEE-33 bus chuẩn
net = nw.case33bw()

# Chạy power flow
pp.runpp(net)

# Xuất dữ liệu bus (điện áp, góc, P, Q)
net.res_bus.to_csv("ieee33_bus.csv", index=True, encoding="utf-8-sig")

# Xuất dữ liệu line (công suất, tổn thất)
net.res_line.to_csv("ieee33_line.csv", index=True, encoding="utf-8-sig")

# Xuất dữ liệu load (P, Q tại các bus)
net.res_load.to_csv("ieee33_load.csv", index=True, encoding="utf-8-sig")

print(">>> Đã xuất dữ liệu IEEE-33 bus ra các file CSV:")
print(" - ieee33_bus.csv")
print(" - ieee33_line.csv")
print(" - ieee33_load.csv")