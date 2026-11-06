import os
import cv2

# ==========================================
# 1. การตั้งค่าโฟลเดอร์
# ==========================================
DATASET_DIR = "dataset"

def create_dirs():
    """สร้างโครงสร้างโฟลเดอร์สำหรับ YOLO"""
    dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    for d in dirs:
        os.makedirs(os.path.join(DATASET_DIR, d), exist_ok=True)
    print("✅ สร้างโฟลเดอร์ images และ labels สำหรับ YOLO เสร็จสิ้น")

def process_data(csv_filename, split_name):
    """
    อ่านไฟล์ CSV (ที่คั่นด้วยช่องว่าง) และสกัดภาพเฟรมกลางจากวิดีโอ
    """
    # ตำแหน่งไฟล์ train.csv หรือ val.csv ของคุณ
    csv_path = os.path.join(DATASET_DIR, "videos", csv_filename) 
    
    if not os.path.exists(csv_path):
        print(f"❌ ไม่พบไฟล์ {csv_path}")
        return
        
    # อ่านไฟล์แบบ Text ธรรมดา เพราะข้อมูลคั่นด้วยช่องว่าง
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        
    print(f"\n🚀 กำลังประมวลผล: {csv_filename} ({len(lines)} วิดีโอ)")
    
    success_count = 0
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # แยกพาร์ทวิดีโอ และ คลาส ID ด้วยช่องว่าง (Space)
        parts = line.split(' ')
        if len(parts) < 2: continue
        
        vid_path = parts[0] # ตัวอย่าง: dataset/videos/Working/work_55787d2e.mp4
        class_id = parts[1] # ตัวอย่าง: 0, 1, 2
        
        # เปิดวิดีโอด้วย OpenCV
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"⚠️ เปิดวิดีโอไม่ได้ (เช็คว่าไฟล์มีจริงไหม): {vid_path}")
            continue
            
        # ดึงเฟรมตรงกลางของวิดีโอมาเป็นภาพตัวแทน
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        
        ret, frame = cap.read()
        if ret:
            # ดึงเฉพาะชื่อไฟล์วิดีโอมาตั้งชื่อรูป (เช่น ได้ work_55787d2e)
            vid_filename = os.path.basename(vid_path) 
            base_name = os.path.splitext(vid_filename)[0] 
            
            # บันทึกรูปภาพ (.jpg)
            img_path = os.path.join(DATASET_DIR, "images", split_name, f"{base_name}.jpg")
            cv2.imwrite(img_path, frame)
            
            # บันทึกไฟล์พิกัด (.txt) 
            # กำหนดให้ Bounding Box คลุมทั้งภาพ (x_center=0.5, y_center=0.5, width=1.0, height=1.0)
            txt_path = os.path.join(DATASET_DIR, "labels", split_name, f"{base_name}.txt")
            with open(txt_path, "w") as f_txt:
                f_txt.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            
            success_count += 1
            
        cap.release()
        
    print(f"✅ ประมวลผล {split_name} เสร็จสิ้น: สกัดภาพสำเร็จ {success_count} รูป")

if __name__ == "__main__":
    # 1. สร้างโฟลเดอร์รองรับ
    create_dirs()
    
    # 2. อ่านไฟล์ train.csv
    process_data("train.csv", "train")
    
    # 3. อ่านไฟล์ val.csv
    process_data("val.csv", "val")