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
from collections import deque, Counter
from ultralytics import YOLO

# ==========================================
# 1. ตั้งค่าพื้นฐานของระบบ
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "gpu")
NUM_CLASSES = 4
IMAGE_SIZE = 224        # ขนาดภาพมาตรฐานของ SlowFast
SEQUENCE_LENGTH = 32    # จำนวนเฟรมที่ใช้เดาท่าทาง 1 ครั้ง
VOTE_LENGTH = 20        # จำนวนครั้งที่ใช้จำผลลัพธ์เพื่อนำมาโหวตหาค่าที่แม่นยำที่สุด

# พิกัดไฟล์วิดีโอ หรือใช้ 0 สำหรับกล้องเว็บแคม
VIDEO_SOURCE = "CLIP/work_01.mp4"  # เปลี่ยนเป็นชื่อไฟล์ได้ เช่น "test_video.mp4"

# ==========================================
# 2. ฟังก์ชันเตรียมข้อมูล (Preprocessing)
# ==========================================
def load_custom_labels():
    """รายชื่อคลาส 4 คลาสของเรา"""
    return {
        0: "Walking",
        1: "Falling",
        2: "Working",
        3: "Not Working"
    }

def preprocess_clip(frames_buffer):
    """ฟังก์ชันแปลงชุดภาพ 32 เฟรม ให้เป็น Tensor รูปแบบ Slow และ Fast"""
    video_np = np.array(frames_buffer)
    clip_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
    
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    clip_tensor = (clip_tensor - mean) / std
    
    fast = clip_tensor.unsqueeze(0)
    slow_idx = torch.linspace(0, fast.shape[2] - 1, fast.shape[2] // 4).long()
    slow = torch.index_select(fast, 2, slow_idx)
    
    return [slow.to(DEVICE), fast.to(DEVICE)]

# ==========================================
# 3. โหลดโมเดล (YOLOv8 & SlowFast)
# ==========================================
print("กำลังโหลด YOLOv8...")
yolo_model = YOLO("yolov8n.pt")

print("กำลังโหลดโมเดล SlowFast แบบ Custom 4 คลาส...")
action_labels = load_custom_labels()

# โหลดโครงสร้างโมเดลและปรับหัวให้มี 4 คลาส
slowfast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
in_features = slowfast_model.blocks[6].proj.in_features
slowfast_model.blocks[6].proj = nn.Linear(in_features, NUM_CLASSES)

# โหลดน้ำหนักที่เทรนเสร็จแล้ว
slowfast_model.load_state_dict(torch.load("custom_slowfast_v1.1.pth", map_location=DEVICE))
slowfast_model = slowfast_model.to(DEVICE).eval()
print("โหลดโมเดลเสร็จสมบูรณ์! พร้อมใช้งาน")

# ==========================================
# 4. ลูปการทำงานหลัก (Main Inference Loop)
# ==========================================
cap = cv2.VideoCapture(VIDEO_SOURCE)

# คิวเก็บภาพ 32 เฟรม ของแต่ละคน
person_buffers = {}
# คิวเก็บผลการทาย 15 ครั้งล่าสุด ของแต่ละคน (สำหรับระบบโหวต)
person_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("จบวิดีโอ หรือไม่สามารถเปิดกล้องได้")
        break

    # ให้ YOLOv8 แกะรอยเฉพาะคน (classes=[0])
    results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            person_resized = cv2.resize(person_rgb, (IMAGE_SIZE, IMAGE_SIZE))

            # สร้างหน่วยความจำให้คนใหม่ที่เพิ่งเดินเข้ากล้อง
            if track_id not in person_buffers:
                person_buffers[track_id] = deque(maxlen=SEQUENCE_LENGTH)
                person_history[track_id] = deque(maxlen=VOTE_LENGTH)

            # เก็บภาพลงคิว
            person_buffers[track_id].append(person_resized)
            
            action_text = "Analyzing..."
            color = (0, 255, 255) # สีเหลืองระหว่างรอข้อมูลครบ
            
            # เมื่อเก็บภาพครบ 32 เฟรม ให้เริ่มทายท่าทาง
            if len(person_buffers[track_id]) == SEQUENCE_LENGTH:
                
                inputs = preprocess_clip(person_buffers[track_id])
                
                with torch.no_grad():
                    outputs = slowfast_model(inputs)
                    pred_idx = outputs.argmax(dim=-1).item()
                    
                # นำผลที่ทายได้รอบนี้ไปใส่ในคิวประวัติ
                person_history[track_id].append(pred_idx)
                
                # +++ ระบบโหวตหาค่าที่แม่นยำที่สุด (Majority Voting) +++
                vote_result = Counter(person_history[track_id]).most_common(1)[0][0]
                
                action_text = action_labels.get(vote_result, "Unknown")
                
                # เปลี่ยนสีตามผลโหวต
                if vote_result == 1: # Falling
                    color = (0, 0, 255) 
                elif vote_result == 2: # Working
                    color = (0, 255, 0)
                elif vote_result == 3: # Not Working
                    color = (0, 165, 255)
                else: # Walking
                    color = (255, 0, 0)

            # วาดกรอบสี่เหลี่ยมและข้อความ
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {action_text}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # แสดงผล
    cv2.imshow("Custom AI Action Recognition", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()