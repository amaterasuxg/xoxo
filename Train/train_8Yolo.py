from ultralytics import YOLO
import torch

# ==========================================
# 1. รายชื่อโมเดลทั้ง 8 ตัว (n และ s)
# ==========================================
models_to_train = [
    'yolov8n.pt', 'yolov8s.pt',
    'yolo11n.pt', 'yolo11s.pt',
    'yolo12n.pt', 'yolo12s.pt',
    'yolo26n.pt', 'yolo26s.pt'  # หากคุณมีไฟล์โมเดลเวอร์ชันนี้ตามตารางที่เคยอ้างอิง
]

# ==========================================
# 2. ตั้งค่าการเทรน (Hyperparameters)
# ==========================================
DATA_YAML = 'data.yaml'
EPOCHS = 50           # จำนวนรอบที่ใช้สอน (แนะนำ 50-100 รอบสำหรับงานจริง)
BATCH_SIZE = 16       # หากการ์ดจอ VRAM ไม่พอแล้วขึ้น Error ให้ปรับลดลงเหลือ 8 หรือ 4
IMG_SIZE = 640
PROJECT_NAME = 'yolo-action-recognition'

# ==========================================
# 3. เริ่มวนลูปเทรนโมเดล
# ==========================================
if __name__ == '__main__':
    for model_name in models_to_train:
        print("="*50)
        print(f"🚀 เริ่มต้นเทรนโมเดล: {model_name}")
        print("="*50)
        
        try:
            # โหลดโมเดลตั้งต้น
            model = YOLO(model_name)
            
            # รันคำสั่ง Train
            # workers=0 ช่วยป้องกันปัญหา Multiprocessing ค้างบน Windows
            results = model.train(
                data=DATA_YAML,
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                workers=0, 
                project=PROJECT_NAME,
                name=f'train_{model_name.replace(".pt", "")}' # ตั้งชื่อโฟลเดอร์เซฟงานตามชื่อโมเดล
            )
            print(f"✅ เทรน {model_name} เสร็จสิ้น!")
            
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดขณะเทรน {model_name}: {e}")
            
    print("\n🎉 เทรนโมเดลครบทั้ง 8 ตัวเรียบร้อยแล้ว!")