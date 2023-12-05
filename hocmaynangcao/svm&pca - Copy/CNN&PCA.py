import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA

# Đường dẫn đến thư mục chứa dữ liệu khuôn mặt
data_folder = "Data_face"

# Kích thước ảnh mà bạn muốn sử dụng để đào tạo mô hình
image_size = (64, 64)

# Tải mô hình đã đào tạo từ tệp "your_model.h5"
model = load_model("your_model.h5")

# Kích hoạt camera
cap = cv2.VideoCapture(0)

# Kích hoạt Haar Cascade Classifier cho việc phát hiện khuôn mặt
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Đọc danh sách tên người và thông tin từ thư mục dữ liệu
class_names = os.listdir(data_folder)
class_info = {}

for class_name in class_names:
    info_path = os.path.join(data_folder, class_name, "info.txt")
    with open(info_path, "r") as file:
        info = file.read()
    class_info[class_name] = info

# Tạo danh sách để lưu trữ các ảnh khuôn mặt và nhãn tương ứng
faces_data = []
labels = []

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 1)
    # Phát hiện khuôn mặt trong khung hình
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Cắt và thay đổi kích thước khuôn mặt để phù hợp với mô hình
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, image_size)

        # Tiền xử lý ảnh trước khi đưa vào mô hình
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        # Sử dụng mô hình để dự đoán
        predictions = model.predict(face)
        prediction_class = int(predictions[0][0] + 0.5)  # Làm tròn kết quả dự đoán

        # Lưu trữ ảnh khuôn mặt và nhãn
        faces_data.append(face.flatten())
        labels.append(prediction_class)

        # Hiển thị kết quả lên màn hình
        if 0 <= prediction_class < len(class_names):
            predicted_name = class_names[prediction_class]
            predicted_info = class_info.get(predicted_name, "No Info")
        else:
            predicted_name = "Unknown"
            predicted_info = "No Info"

        cv2.putText(frame, f"Info: {predicted_info}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # cuoi
        smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)
        for x, y, w, h in smile:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

    # Hiển thị khung hình
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đóng camera và cửa sổ
cap.release()
cv2.destroyAllWindows()

# Chuyển đổi danh sách ảnh khuôn mặt và nhãn thành numpy arrays
faces_data = np.array(faces_data)
labels = np.array(labels)

# Sử dụng PCA để giảm chiều dữ liệu
n_components = 50  # Số thành phần chính bạn muốn giữ lại
pca = PCA(n_components=n_components)
faces_data_pca = pca.fit_transform(faces_data)
