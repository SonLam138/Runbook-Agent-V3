from Call_V4_nho import run_v4_nho

result = run_v4_nho(f"D:\Temp\Lỗi truy cập T24.eml")

print("STATUS:", result["status"])
print("---------")
print("LOG SUMMARY:")
print(result["log_summary"])