import os
import cv2
from ultralytics import YOLO

# ==========================================
# 1. ตั้งค่าตำแหน่งไฟล์ (แก้ให้ตรงกับเครื่องของคุณ)
# ==========================================
# เลือกไฟล์ best.pt จากโมเดลที่ทำคะแนนได้ดีที่สุด (เช่น yolo26n หรือ yolo11n)
MODEL_PATH = "runs/detect/yolo-action-recognition/train_yolo26n/weights/best.pt"

# ตำแหน่งไฟล์คลิปวิดีโอที่คุณต้องการทดสอบ (เช่น คลิปจากชุดข้อมูล UCF101)
VIDEO_PATH = "CLIP/work_01.mp4"  # ใส่ 0 หากต้องการใช้เว็บแคม

def main():
    # ==========================================
    # 2. ตรวจสอบว่ามีไฟล์อยู่จริงหรือไม่
    # ==========================================
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: ไม่พบไฟล์โมเดลที่ -> {MODEL_PATH}")
        return
        
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Error: ไม่พบไฟล์วิดีโอที่ -> {VIDEO_PATH}")
        return

    print(f"\n🚀 โหลดโมเดล: {MODEL_PATH}")
    print(f"📹 กำลังทดสอบวิดีโอ: {VIDEO_PATH}")
    print("="*50)

    # ==========================================
    # 3. โหลดโมเดลและสั่งรันตรวจจับ
    # ==========================================
    model = YOLO(MODEL_PATH)
    
    # รัน Predict
    results = model.predict(
        source=VIDEO_PATH,
        save=True,       # บันทึกวิดีโอผลลัพธ์ (จะถูกเซฟใน runs/detect/predict)
        show=True,       # แสดงหน้าต่างวิดีโอขณะรันตรวจจับ (ถ้าเครื่องรองรับ UI)
        conf=0.5,        # ความมั่นใจขั้นต่ำ 50% (ปรับลดลงได้ถ้ามันตีกรอบยาก)
        line_width=2     # ความหนาของเส้น Bounding Box
    )

    print("\n" + "="*50)
    print("✅ ตรวจจับเสร็จสิ้น!")
    
    # ดึงตำแหน่งที่บันทึกไฟล์วิดีโอผลลัพธ์ออกมาแสดง
    save_dir = results[0].save_dir
    print(f"📁 วิดีโอผลลัพธ์ถูกบันทึกไว้ที่: {save_dir}")

if __name__ == '__main__':
    main()