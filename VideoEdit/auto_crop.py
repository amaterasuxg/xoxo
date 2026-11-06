import os
import cv2
import pandas as pd
from ultralytics import YOLO

# ==========================================
# 1. ตั้งค่าโฟลเดอร์และไฟล์
# ==========================================
YOLO_MODEL = 'yolov8n.pt'  # โมเดลพื้นฐานสำหรับหา "คน"
CSV_PATH = 'dataset/val.csv'
OUTPUT_DIR = 'dataset/slowfast_data/val' # โฟลเดอร์ปลายทางที่จะเซฟรูป

# กำหนดชื่อคลาสตาม Label ของคุณ (0, 1, 2)
CLASSES = {0: 'Working', 1: 'Not_Working', 2: 'Falling'}

def process_video(video_path, label, video_id, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ เปิดวิดีโอไม่ได้: {video_path}")
        return

    cls_name = CLASSES.get(label, "Unknown")
    
    # สร้างโฟลเดอร์ย่อยสำหรับวิดีโอนี้ (เช่น Working/video_001/)
    vid_out_dir = os.path.join(OUTPUT_DIR, cls_name, f"video_{video_id:04d}")
    os.makedirs(vid_out_dir, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # รัน YOLO จับเฉพาะคลาส 0 (คน)
        results = model.predict(frame, classes=[0], verbose=False, conf=0.5)
        
        # ถ้าเจอคนอย่างน้อย 1 คน
        if len(results[0].boxes) > 0:
            # ดึงกล่องที่โมเดลมั่นใจที่สุดมาใช้งาน
            best_box = results[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, best_box)
            
            # ป้องกันกรอบทะลุขอบภาพ
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # ตัด (Crop) เฉพาะภาพคน
            cropped_person = frame[y1:y2, x1:x2]
            
            # กรองภาพที่เล็กเกินไปทิ้ง (Noise)
            if cropped_person.shape[0] < 50 or cropped_person.shape[1] < 50:
                continue

            # เซฟรูปเก็บไว้เรียงตามเฟรม
            save_path = os.path.join(vid_out_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(save_path, cropped_person)
            saved_count += 1

    cap.release()
    print(f"✅ คลิป {video_id} ({cls_name}): ตัดภาพคนได้ {saved_count} เฟรม")

def main():
    print(f"🚀 กำลังโหลดโมเดล YOLO: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)
    
    print(f"📂 กำลังอ่านไฟล์ CSV: {CSV_PATH}")
    # ** จุดต้องแก้: ให้ตรวจสอบว่าใน CSV ของคุณ หัวคอลัมน์ชื่ออะไร **
    # สมมติว่าคอลัมน์ชื่อ 'video_path' และ 'label'
    df = pd.read_csv(CSV_PATH, header=None, names=['video_path', 'label'], sep=' ') 
    
    # สร้างโฟลเดอร์หลักเตรียมไว้
    for cls_name in CLASSES.values():
        os.makedirs(os.path.join(OUTPUT_DIR, cls_name), exist_ok=True)
        
    # วนลูปอ่านวิดีโอทีละไฟล์
    for idx, row in df.iterrows():
        # ถ้าคอลัมน์ของคุณชื่ออื่น ให้เปลี่ยนคำว่า 'video_path' และ 'label' ให้ตรงกัน
        process_video(row['video_path'], row['label'], idx, model)
        
    print("\n🎉 สกัดภาพคนเสร็จสมบูรณ์! ข้อมูลพร้อมสำหรับเทรน SlowFast แล้ว")

if __name__ == '__main__':
    main()