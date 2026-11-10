# ==========================================
# 0. โค้ดแก้ปัญหาไลบรารี pytorchvideo
# ==========================================
import sys
import torchvision.transforms.functional as F_vision
sys.modules['torchvision.transforms.functional_tensor'] = F_vision

import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. ตั้งค่าและโหลดโมเดล
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
CLASSES = ["Working", "Not Working", "Falling"]
COLORS = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]

# โหลด YOLOv12
print("กำลังโหลดโมเดล YOLOv12...")
yolo_model = YOLO("yolo12s.pt") 

# โหลด X3D
print("กำลังโหลดโมเดล X3D...")
x3d_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=False)
# X3D สมองอยู่ block 5
in_features = x3d_model.blocks[5].proj.in_features
x3d_model.blocks[5].proj = nn.Linear(in_features, NUM_CLASSES)

# โหลดไฟล์น้ำหนักโมเดล X3D ที่เทรนไว้
x3d_model.load_state_dict(torch.load("custom_x3d_3class_run50.pth", map_location=DEVICE))
x3d_model = x3d_model.to(DEVICE)
x3d_model.eval()

# ==========================================
# 2. ฟังก์ชันเตรียมข้อมูล (ส่งก้อนเดียว)
# ==========================================
def prepare_x3d_input(frame_buffer, image_size=224):
    video_np = np.array(frame_buffer)
    # แปลงเป็น Tensor (C, T, H, W)
    clip_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
    
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    clip_tensor = (clip_tensor - mean) / std
    
    # ส่งออกโดยเพิ่มมิติ Batch (1, C, T, H, W)
    return clip_tensor.unsqueeze(0).to(DEVICE)

# ==========================================
# 3. รันวิดีโอและทำนายผล
# ==========================================
video_path = "CLIP/sit1.mp4" 
cap = cv2.VideoCapture(video_path)

frame_buffer = []
current_action = "Waiting..."
current_color = (255, 255, 255)

print("🚀 เริ่มการตรวจจับ (YOLOv12 + X3D)...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    # 1. YOLO หาตำแหน่งคน
    results = yolo_model.predict(frame, classes=[0], verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 2. ตัดภาพคน
        cropped_person = frame[y1:y2, x1:x2]
        if cropped_person.size > 0:
            person_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
            person_resized = cv2.resize(person_rgb, (224, 224))
            frame_buffer.append(person_resized)
            
            # 3. ครบ 32 เฟรม ให้ X3D ทายท่าทาง
            if len(frame_buffer) == 32:
                inputs = prepare_x3d_input(frame_buffer)
                with torch.no_grad():
                    preds = x3d_model(inputs)
                    predicted_class = torch.argmax(preds, dim=1).item()
                    
                current_action = CLASSES[predicted_class]
                current_color = COLORS[predicted_class]
                
                # เลื่อน Buffer ออกครึ่งหนึ่ง
                frame_buffer = frame_buffer[16:]
                
        # 4. วาดกรอบบนจอ
        cv2.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
        cv2.putText(frame, current_action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_color, 2)
    else:
        frame_buffer.clear()
        current_action = "No Person"
        
    cv2.imshow("YOLOv12 + X3D", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()