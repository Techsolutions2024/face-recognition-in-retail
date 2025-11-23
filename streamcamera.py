import cv2
import requests
import numpy as np

url = "http://192.168.1.44:4747/video"

# Mở stream dạng MJPEG
try:
    stream = requests.get(url, stream=True, timeout=5)
    print("Kết nối thành công tới camera IP!")
except Exception as e:
    print("Không thể kết nối tới URL:", e)
    exit()

bytes_data = b''

while True:
    # Đọc dữ liệu từ stream
    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk

        # Tìm byte bắt đầu và kết thúc của ảnh JPEG
        start = bytes_data.find(b'\xff\xd8')  # SOI (Start of image)
        end   = bytes_data.find(b'\xff\xd9')  # EOI (End of image)

        # Khi tìm thấy 1 frame đầy đủ
        if start != -1 and end != -1:
            jpg = bytes_data[start:end+2]
            bytes_data = bytes_data[end+2:]  # cắt bỏ phần đã xử lý

            # Giải mã JPEG thành frame
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is not None:
                cv2.imshow("IP Camera Stream", frame)

            # Nhấn Q để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit()
