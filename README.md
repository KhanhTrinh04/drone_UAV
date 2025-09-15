# drone_UAV

**Repository:** `KhanhTrinh04/drone_UAV`  
**Ngôn ngữ:** Python  
**Mục đích:** Hệ thống phát hiện/vị trí (detection) mục tiêu từ ảnh/video drone (UAV).

---

## 📌 Tổng quan
`drone_UAV` là kho chứa mã nguồn cho bài toán phát hiện đối tượng trên ảnh/video thu từ drone.  
Mục tiêu có thể gồm: phát hiện buồng chuối (banana bunch), người, phương tiện, hoặc các target khác tùy dataset.  
Kho này hướng tới cả nghiên cứu (train/finetune) và triển khai inference real-time.

---

## ✨ Tính năng
- Huấn luyện/finetune mô hình phát hiện (YOLO/Ultralytics).  
- Chạy inference trên ảnh đơn, folder ảnh, video và stream real-time (webcam/RTSP).  
- Script tiện ích để convert/đóng gói mô hình và xuất bounding boxes + confidence.  
- Dataset format chuẩn YOLO (txt) và có hỗ trợ convert từ XML (VOC).  

---

## 📂 Cấu trúc repo
drone_UAV/
├── data/
│ ├── images/
│ │ ├── train/
│ │ ├── val/
│ │ └── test/
│ └── labels/
│ ├── train/
│ ├── val/
│ └── test/
├── configs/
│ └── yolovX_custom.yaml
├── scripts/
│ ├── train.py
│ ├── detect.py
│ └── realtime.py
├── utils/
│ ├── dataset_converter.py
│ └── viz.py
├── runs/ # output training / inference
├── requirements.txt
└── README.md


---

## ⚙️ Cài đặt
Yêu cầu:
- Python 3.8+  
- CUDA (nếu dùng GPU)  
- Thư viện: `ultralytics`, `torch`, `opencv-python`, `numpy`, `tqdm`, `pyyaml`, `pillow`

Cài đặt nhanh:
```bash
python -m venv venv
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate        # Windows
pip install -r requirements.txt
📊 Dataset

YOLO TXT: <class> <x_center> <y_center> <width> <height> (giá trị [0,1])

XML VOC → YOLO TXT: dùng utils/dataset_converter.py

Ví dụ data/dataset.yaml:

train: data/images/train
val: data/images/val
test: data/images/test

nc: 1
names: ['banana']

🚀 Huấn luyện

Sử dụng Ultralytics CLI:

yolo detect train data=data/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640


Hoặc script:

python scripts/train.py --data data/dataset.yaml --cfg configs/yolovX_custom.yaml --weights yolov8n.pt --epochs 100

🔍 Inference

Trên folder ảnh:

yolo detect predict model=runs/detect/train/weights/best.pt source="data/images/val" save=True


Trên 1 ảnh bằng Python:

from ultralytics import YOLO

model = YOLO(r"C:/path/to/runs/train/weights/best.pt")
results = model.predict(source=r"C:/path/to/image.jpg", show=True)

🎥 Real-time

Webcam:

python scripts/realtime.py --weights runs/train/weights/best.pt --source 0


RTSP:

python scripts/realtime.py --weights runs/train/weights/best.pt --source "rtsp://user:pass@ip:554/stream"

📈 Đánh giá
yolo val model=runs/train/weights/best.pt data=data/dataset.yaml

🔄 Export mô hình

Ví dụ export sang ONNX:

yolo export model=runs/train/weights/best.pt format=onnx

🛠️ Tips & Troubleshooting

Lỗi CUDA → kiểm tra version torch và driver GPU.

Model không generalize → kiểm tra dataset balance, augmentation, label quality.

Tránh overfit → early stopping, tăng augmentation, regularization.

🤝 Contributing

Fork repo → tạo branch: feature/your-feature

Commit, push và mở Pull Request

Viết rõ mô tả, cách test, và file chỉnh sửa

📜 License

MIT License © 2025 KhanhTrinh04
