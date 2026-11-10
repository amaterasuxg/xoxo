# ==========================================
# 0. โค้ดแก้ปัญหาไลบรารี pytorchvideo (ต้องอยู่บนสุด)
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "gpu")

# ตั้งค่า 3 คลาส
NUM_CLASSES = 3
CLASSES = ["Working", "Not Working", "Falling"]
COLORS = [(0, 255, 0), (0, 165, 255), (0, 0, 255)] # สีเขียว, สีส้ม, สีแดง (BGR)

# --- 1.1 โหลด YOLO (อัปเดต Path ให้ตรงกับโฟลเดอร์ใหม่ของคุณ) ---
print("กำลังโหลดโมเดล YOLO (ค้นหาคน)...")
# เปลี่ยนชื่อโฟลเดอร์ train_yolo... ให้ตรงกับตัวที่คุณอยากใช้
yolo_model = YOLO("yolo11n.pt")

# --- 1.2 โหลด SlowFast (วิเคราะห์ท่าทาง) ---
print("กำลังโหลดโมเดล SlowFast (วิเคราะห์ท่าทาง)...")
slowfast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
in_features = slowfast_model.blocks[6].proj.in_features
slowfast_model.blocks[6].proj = nn.Linear(in_features, NUM_CLASSES)

# โหลดไฟล์น้ำหนักที่คุณเทรนมา (แก้ชื่อไฟล์ .pth ให้ตรงกับของคุณที่มีในโฟลเดอร์)
slowfast_model.load_state_dict(torch.load("Trained/SlowFast.pth", map_location=DEVICE))
slowfast_model = slowfast_model.to(DEVICE)
slowfast_model.eval()

# ==========================================
# 2. ฟังก์ชันเตรียมข้อมูลให้ SlowFast
# ==========================================
def prepare_slowfast_input(frame_buffer):
    video_np = np.array(frame_buffer)
    # แปลงข้อมูลให้อยู่ในรูปแบบที่ PyTorchVideo ต้องการ (C, T, H, W)
    clip_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
    
    # ปรับค่าสีตามมาตรฐาน (Normalize)
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    clip_tensor = (clip_tensor - mean) / std
    
    # แยกเป็นเส้นทางเร็ว (Fast 32 เฟรม) และเส้นทางช้า (Slow 8 เฟรม)
    fast = clip_tensor
    slow_idx = torch.linspace(0, fast.shape[1] - 1, fast.shape[1] // 4).long()
    slow = torch.index_select(fast, 1, slow_idx)
    
    return [slow.unsqueeze(0).to(DEVICE), fast.unsqueeze(0).to(DEVICE)]

# ==========================================
# 3. รันวิดีโอและทำนายผล
# ==========================================
video_path = "TestClip/work_02.mp4" # เปลี่ยนเป็น 0 หากต้องการใช้กล้องเว็บแคม
cap = cv2.VideoCapture(video_path)

frame_buffer = []
current_action = "Waiting..."
current_color = (255, 255, 255)

# ตัวแปรสำหรับระบบโหวต (Temporal Voting) ลดอาการสวิงของกรอบข้อความ
prediction_history = []
VOTE_SIZE = 7

print("🚀 เริ่มการตรวจจับแบบ Hybrid (YOLO + SlowFast 3 Classes)...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ วิดีโอเล่นจบแล้ว")
        break
        
    # --- ขั้นตอนที่ 1: ค้นหาคนด้วย YOLO ---
    results = yolo_model.predict(frame, classes=[0], verbose=False) # classes=[0] คือหาเฉพาะคน
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) > 0:
        # ใช้คนแรกที่ตรวจจับเจอ (index 0)
        x1, y1, x2, y2 = map(int, boxes[0])
        h, w, _ = frame.shape
        
        box_w = x2 - x1
        box_h = y2 - y1
        
        # ขยายกรอบ Bounding Box ออกไป 15% เพื่อให้เห็นท่าทางชัดขึ้นและกันกรอบสั่น
        pad_x = int(box_w * 0.15)
        pad_y = int(box_h * 0.15)
        px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        px2, py2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        cropped_person = frame[py1:py2, px1:px2]
        
        # --- ขั้นตอนที่ 2: สะสมภาพเพื่อส่งให้ SlowFast ---
        if cropped_person.size > 0:
            person_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
            person_resized = cv2.resize(person_rgb, (224, 224))
            frame_buffer.append(person_resized)
            
            # เมื่อเก็บครบ 32 เฟรม ให้เริ่มทำนายท่าทาง
            if len(frame_buffer) == 32:
                inputs = prepare_slowfast_input(frame_buffer)
                
                with torch.no_grad():
                    preds = slowfast_model(inputs)
                    
                    probs = torch.nn.functional.softmax(preds, dim=1)
                    max_prob, predicted_class_tensor = torch.max(probs, dim=1)
                    
                    predicted_class = predicted_class_tensor.item()
                    confidence = max_prob.item()
                            
                # ระบบโหวตเสียงข้างมากจาก 7 เฟรมล่าสุด
                prediction_history.append(predicted_class)
                if len(prediction_history) > VOTE_SIZE:
                    prediction_history.pop(0)
                    
                final_voted_class = max(set(prediction_history), key=prediction_history.count)
                
                current_action = f"{CLASSES[final_voted_class]} ({confidence*100:.1f}%)"
                current_color = COLORS[final_voted_class]
                
                # เลื่อน Buffer ออกครึ่งหนึ่ง (Sliding Window) เพื่อความลื่นไหล
                frame_buffer = frame_buffer[16:]
                
        # วาดกรอบและแสดงข้อความบนจอ
        cv2.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
        cv2.putText(frame, current_action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)
    else:
        # ถ้าไม่เจอคน ให้ล้างข้อมูลทิ้ง
        frame_buffer.clear()
        prediction_history.clear()
        current_action = "No Person"
        current_color = (255, 255, 255)
        
    cv2.imshow("Hybrid Fall Detection (YOLO + SlowFast 3 Classes)", frame)
    
    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()