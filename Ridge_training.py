import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np
#Doc du lieu
df = pd.read_csv("bmi.csv")
print(df.head())
#Chia du lieu
X = df[["Chieu cao", "Can nang"]]
y = df["chi so BMI"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Khởi tạo mô hình ridge
ridge_model = Ridge(alpha=1.0)

#Huấn luyện mô hình
ridge_model.fit(X_train,y_train)
#Dự đoán tập kiểm tra
y_pred = ridge_model.predict(X_test)

#
r_squared = r2_score(y_test,y_pred)
print("Hệ số R²:", r_squared)

mae = mean_absolute_error(y_test,y_pred)
print("MAE:", mae)

rmse =  np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

def calculate_nse(y_test, y_pred):
    numerator = np.sum((y_test - y_pred) ** 2)
    denominator = np.sum((y_test - np.mean(y_test)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

nse = calculate_nse(y_test, y_pred)
print("NSE:", nse)
