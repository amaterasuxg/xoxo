import os
import pandas as pd
import wandb
from ultralytics import YOLO

DATA_YAML_PATH = "data.yaml"
WANDB_PROJECT = "yolo-action-recognition"
WANDB_RUN_NAME = "compare_yolo_models_final"
IMG_SIZE = 640

# ==========================================
# 2. ระบุพาร์ทของไฟล์โมเดลที่เทรนเสร็จแล้ว 
# (อ้างอิงจากโฟลเดอร์ runs/detect/yolo-action-recognition ของคุณ)
# ==========================================
MODELS_TO_TEST = [
    'runs/detect/yolo-action-recognition/train_yolov8n/weights/best.pt',
    'runs/detect/yolo-action-recognition/train_yolov8s/weights/best.pt',
    'runs/detect/yolo-action-recognition/train_yolo11n/weights/best.pt',
    'runs/detect/yolo-action-recognition/train_yolo11s/weights/best.pt',
    'runs/detect/yolo-action-recognition/train_yolo12n/weights/best.pt',
    'runs/detect/yolo-action-recognition/train_yolo12s/weights/best.pt',
    'runs/detect/yolo-action-recognition/train_yolo26n/weights/best.pt',
    'runs/detect/yolo-action-recognition/train_yolo26s/weights/best.pt'
]

def main():
    # ตรวจสอบไฟล์ data.yaml
    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ Error: ไม่พบไฟล์ '{DATA_YAML_PATH}'")
        return

    # เริ่มเชื่อมต่อ wandb
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
    table_data = []

    # วนลูปทดสอบโมเดลทีละตัว
    for model_path in MODELS_TO_TEST:
        # เช็คว่าไฟล์โมเดลมีอยู่จริงไหม (ถ้าไม่มีจะปริ้นท์บอกพาร์ทที่หาไม่เจอ)
        if not os.path.exists(model_path):
            print(f"⚠️ ข้ามโมเดล: หาไฟล์ไม่เจอที่ -> {model_path}")
            continue

        print(f"\n🚀 กำลังประเมินโมเดล: {model_path}...")
        
        # โหลดโมเดล
        model = YOLO(model_path)
        
        # รัน Validation (workers=0 แก้ปัญหาค้างบน Windows)
        results = model.val(data=DATA_YAML_PATH, imgsz=IMG_SIZE, workers=0)
        
        # ดึงค่าสถิติ
        map50_95 = results.box.map * 100
        speed_ms = results.speed['inference']
        
        # ดึงค่า params และ flops (รับค่า 4 ตัวป้องกัน Error Unpack)
        layers, params, gradients, flops = model.info()
        params_m = params / 1e6
        
        # สกัดชื่อโมเดลสั้นๆ จากพาร์ทเพื่อความสวยงามในตาราง (เช่น 'yolov8n')
        short_name = model_path.split('/')[-3].replace('train_', '')
        
        # เก็บข้อมูลลงตาราง
        table_data.append({
            "Model": short_name,
            "Size (pixels)": IMG_SIZE,
            "mAP 50-95": round(map50_95, 1),
            "Speed (ms)": round(speed_ms, 2),
            "Params (M)": round(params_m, 2),
            "FLOPs (B)": round(flops, 1)
        })

    # ตรวจสอบข้อมูลก่อนสร้างตาราง (Debug)
    print("\n--- เช็คข้อมูลดิบที่ดึงมาได้ ---")
    print(table_data)
    print("--------------------------------\n")

    # ==========================================
    # 3. แสดงผลตารางและอัปโหลดขึ้น wandb
    # ==========================================
    if table_data:
        df = pd.DataFrame(table_data)
        print(" 📊 ตารางเปรียบเทียบประสิทธิภาพโมเดล")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50 + "\n")
        
        # อัปโหลดตารางขึ้น wandb
        wandb_table = wandb.Table(dataframe=df)
        wandb.log({"Model_Comparison": wandb_table})
        print("✅ อัปโหลดตารางขึ้น wandb เรียบร้อยแล้ว! กดคลิกลิงก์ด้านล่างเพื่อดูผลได้เลย")
    else:
        print("❌ ไม่มีข้อมูลสำหรับสร้างตาราง (กรุณาเช็คว่ามีไฟล์ best.pt อยู่จริงตามโฟลเดอร์ที่แจ้งเตือนไว้ด้านบนหรือไม่)")

    # ปิดการเชื่อมต่อ
    wandb.finish()

# รันฟังก์ชันหลัก (ป้องกัน Error Multiprocessing)
if __name__ == '__main__':
    main()