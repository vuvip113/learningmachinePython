import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Bước 1: Chuẩn bị Dữ liệu
# (Bao gồm thu thập dữ liệu và tiền xử lý)

# Bước 2: Xây dựng Mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # 1 lớp đầu ra cho phân loại nhận diện hoặc không nhận diện
])

# Bước 3: Biên Dịch Mô Hình
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Bước 4: Huấn Luyện Mô Hình
# (Sử dụng tập dữ liệu đã chuẩn bị)

# Bước 5: Lưu Mô Hình
model.save("your_model.h5")
