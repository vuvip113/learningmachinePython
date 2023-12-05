import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('forestfires.csv')
print(data)
# Chọn các features để làm đầu vào cho mô hình
features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

# # Tạo ma trận X chứa các features và vector y chứa biến mục tiêu (diện tích cháy rừng)
X = data[features].values
y = data['area'].values

print(X)
print(y)
# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Thực hiện PCA để giảm chiều dữ liệu
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(X_train_pca)
print(X_test_pca)
# Huấn luyện mô hình hồi quy tuyến tính trên dữ liệu sau khi giảm chiều
model = LinearRegression()
model.fit(X_train_pca, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_pca)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)