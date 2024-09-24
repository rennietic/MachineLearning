import pandas as pd
from sklearn.model_selection import train_test_split  # chia dữ liệu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("bmi.csv")  # Đường dẫn tới file dữ liệu

# Hiển thị vài hàng đầu tiên
print(df.head())

# Chia dữ liệu
X = df[["Chieu cao", "Can nang"]]
y = df["chi so BMI"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hồi quy tuyến tính
model = LinearRegression()  # khởi tạo mô hình hồi quy tuyến tính
model.fit(X_train, y_train)  # Bắt đầu huấn luyện

# Dự đoán
y_pred = model.predict(X_test)
# Thống kê
print("Dự đoán:", y_pred[:5])
print("Giá trị thực tế:", y_test[:5].values)

# Đánh giá
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

r_squared = r2_score(y_test, y_pred)
print("Hệ số R2:", r_squared)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# Tính NSE (Nash-Sutcliffe Efficiency)
def calculate_nse(y_test, y_pred):
    numerator = np.sum((y_test - y_pred) ** 2)
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

nse = calculate_nse(y_test, y_pred)
print("NSE:", nse)

# Vẽ biểu đồ
plt.figure(figsize=(15, 5))

# Biểu đồ hồi quy tuyến tính
plt.subplot(1, 1, 1)  # Chỉ có một biểu đồ
plt.scatter(y_test, y_pred, color='blue', label='Dữ liệu thực tế', s=30)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Đường hoàn hảo')
plt.title('Hồi quy tuyến tính')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.legend()
plt.grid(True)

plt.show()
