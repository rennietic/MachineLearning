import pandas as pd
from sklearn.model_selection import train_test_split #chia dl
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import numpy as np
# df = pd.read_csv("d:/Nam3/MachineLearning/Code/duan1/bmi.csv")
df = pd.read_csv("bmi.csv")

# Display the first few rows
print(df.head())

#Chia dữ liệu 
X = df[["Chieu cao","Can nang"]]
y = df["chi so BMI"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

#Hồi quy tuyến tính
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42) #khởi tạo mô hình 
mlp_model.fit(X_train,y_train) #Bắt đầu huấn luyện

#du doan
y_pred = mlp_model.predict(X_test) 
#thu
print("Dự đoán:", y_pred[:5])
print("Giá trị thực tế:", y_test[:5].values)

#danh gia
mae = mean_absolute_error(y_test,y_pred)
print("MAE:", mae)

#danh gia r2
r_squared = r2_score(y_test,y_pred)
print("Hệ số R2:",r_squared)

#danh gia RMSE
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("RMSE:", rmse)

# Tính NSE (Nash-Sutcliffe Efficiency)
def calculate_nse(y_test, y_pred):
    numerator = np.sum((y_test - y_pred) ** 2)
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

nse = calculate_nse(y_test, y_pred)
print("NSE:", nse)