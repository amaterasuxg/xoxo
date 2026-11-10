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
import threading
from ultralytics import YOLO

# ==========================================
# 1. คลาสสำหรับอ่านกล้อง IP แบบ Real-time (แยก Thread ป้องกันภาพดีเลย์)
# ==========================================
class RealtimeCamera:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True

# ==========================================
# 2. ตั้งค่าและโหลดโมเดล
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3
CLASSES = ["Working", "Not Working", "Falling"]
COLORS = [(0, 255, 0), (0, 165, 255), (0, 0, 255)] 

print("กำลังโหลดโมเดล YOLO (ค้นหาคน)...")
yolo_model = YOLO("yolo11n.pt")

print("กำลังโหลดโมเดล SlowFast (วิเคราะห์ท่าทาง)...")
slowfast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
in_features = slowfast_model.blocks[6].proj.in_features
slowfast_model.blocks[6].proj = nn.Linear(in_features, NUM_CLASSES)
slowfast_model.load_state_dict(torch.load("trained/custom_slowfast_3class_best.pth", map_location=DEVICE))
slowfast_model = slowfast_model.to(DEVICE)
slowfast_model.eval()

# ==========================================
# 3. ฟังก์ชันเตรียมข้อมูลให้ SlowFast
# ==========================================
def prepare_slowfast_input(frame_buffer):
    video_np = np.array(frame_buffer)
    clip_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
    clip_tensor = (clip_tensor - mean) / std
    
    fast = clip_tensor
    slow_idx = torch.linspace(0, fast.shape[1] - 1, fast.shape[1] // 4).long()
    slow = torch.index_select(fast, 1, slow_idx)
    return [slow.unsqueeze(0).to(DEVICE), fast.unsqueeze(0).to(DEVICE)]

# ==========================================
# 4. รันวิดีโอและทำนายผล
# ==========================================
# ⚠️ เปลี่ยนตรงนี้เป็นลิงก์ RTSP หรือ HTTP ของกล้อง Wi-Fi ของคุณ
video_path = "rtsp://admin:group166@192.168.1.64:554/Streaming/Channels/101" 

print("กำลังเชื่อมต่อกล้อง...")
cap = RealtimeCamera(video_path).start()

frame_buffer = []
current_action = "Waiting..."
current_color = (255, 255, 255)
prediction_history = []
VOTE_SIZE = 7

# --- ตัวแปรสำหรับปรับแต่งความลื่นไหล ---
frame_count = 0
SKIP_FRAMES = 2  # ประมวลผล 1 เฟรม ข้าม 2 เฟรม (ลดภาระเครื่องได้ 3 เท่า)

# --- ตัวแปรสำหรับแก้กรอบวาร์ป (Smooth Bounding Box) ---
smooth_box = None
ALPHA = 0.4  # ค่าความสมูท (0.1 ถึง 1.0) ถ้าน้อย = กรอบไหลลื่นวิ่งตามช้า / ถ้ามาก = ตามไวแต่แข็ง
sx1 = sy1 = sx2 = sy2 = 0
has_person = False

print("🚀 เริ่มการตรวจจับแบบ Hybrid (Real-time Threading & Smooth Box)...")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue # ถ้าดึงภาพไม่ได้ ให้รอรอบถัดไป
        
    frame_count += 1
    
    # --- รัน AI เฉพาะเฟรมที่กำหนด (ข้ามเฟรมเพื่อลดภาระ) ---
    if frame_count % SKIP_FRAMES == 0:
        results = yolo_model.predict(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0:
            has_person = True
            x1, y1, x2, y2 = map(int, boxes[0])
            
            # --- ระบบคำนวณสมูทกรอบ (EMA) ---
            current_box = np.array([x1, y1, x2, y2])
            if smooth_box is None:
                smooth_box = current_box # เจอคนครั้งแรกให้กรอบอยู่ที่คนเลย
            else:
                smooth_box = (ALPHA * current_box) + ((1 - ALPHA) * smooth_box)
                
            sx1, sy1, sx2, sy2 = map(int, smooth_box)
            
            # ใช้พิกัดดั้งเดิมของเฟรมนั้นๆ เพื่อความแม่นยำในการตัดภาพ
            h, w, _ = frame.shape
            box_w, box_h = x2 - x1, y2 - y1
            pad_x, pad_y = int(box_w * 0.15), int(box_h * 0.15)
            px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            px2, py2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            
            cropped_person = frame[py1:py2, px1:px2]
            
            if cropped_person.size > 0:
                person_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
                person_resized = cv2.resize(person_rgb, (224, 224))
                frame_buffer.append(person_resized)
                
                # เมื่อเก็บภาพครบ 32 เฟรม
                if len(frame_buffer) == 32:
                    inputs = prepare_slowfast_input(frame_buffer)
                    
                    with torch.no_grad():
                        preds = slowfast_model(inputs)
                        probs = torch.nn.functional.softmax(preds, dim=1)
                        max_prob, predicted_class_tensor = torch.max(probs, dim=1)
                        
                        predicted_class = predicted_class_tensor.item()
                        confidence = max_prob.item()
                                
                    # ระบบโหวต
                    prediction_history.append(predicted_class)
                    if len(prediction_history) > VOTE_SIZE:
                        prediction_history.pop(0)
                        
                    final_voted_class = max(set(prediction_history), key=prediction_history.count)
                    current_action = f"{CLASSES[final_voted_class]} ({confidence*100:.1f}%)"
                    current_color = COLORS[final_voted_class]
                    
                    # Sliding Window
                    frame_buffer = frame_buffer[16:]
        else:
            # ไม่เจอคน
            has_person = False
            smooth_box = None
            frame_buffer.clear()
            prediction_history.clear()
            current_action = "Waiting..."
            current_color = (255, 255, 255)
            
    # --- วาดกรอบและแสดงผล (วาดบนทุกเฟรม ไม่ว่า AI จะทำงานข้ามเฟรมอยู่หรือไม่) ---
    if has_person:
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), current_color, 2)
        cv2.putText(frame, current_action, (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

    cv2.imshow("Hybrid Fall Detection (Real-time & Smooth Box)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()