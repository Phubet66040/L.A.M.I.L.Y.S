import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# สร้างข้อมูลตัวอย่าง (X = ค่าอินพุต, y = ผลลัพธ์)
X = np.array([[15], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # ค่าอินพุต
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # ผลลัพธ์ที่คาดหวัง

# สร้างโมเดล Linear Regression
model = LinearRegression()

# ฝึกโมเดล
model.fit(X, y)

# ทำนายผลสำหรับข้อมูลใหม่
X_new = np.array([[11], [12], [13]])  # ข้อมูลใหม่ที่ต้องการทำนาย
y_pred = model.predict(X_new)

# แสดงผลการทำนาย
print("Predictions:", y_pred)

# แสดงกราฟ
plt.scatter(X, y, color='blue')  # จุดข้อมูลจริง
plt.plot(X, model.predict(X), color='red')  # เส้นที่โมเดลทำนาย
plt.xlabel('Input (X)')
plt.ylabel('Output (y)')
plt.title('Linear Regression Example')
plt.show()
