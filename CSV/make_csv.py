import os
import glob
import random

# 1. ตั้งค่าโฟลเดอร์เก็บวิดีโอและ Label (แก้ชื่อโฟลเดอร์ให้ตรงกับของเครื่องคุณ)
# สมมติว่าคุณแยกวิดีโอไว้ใน 3 โฟลเดอร์ตามท่าทาง
DATA_FOLDERS = {
    "Dataset/video/Working": 0,
    "Dataset/video/Not_Working": 1,
    "Dataset/video/Falling": 2
}

all_videos = []

# 2. ค้นหาทั้งไฟล์ .mp4 และ .avi
for folder, label in DATA_FOLDERS.items():
    if not os.path.exists(folder):
        print(f"ไม่พบโฟลเดอร์: {folder}")
        continue
        
    # ดึงไฟล์ทั้งหมด
    videos = glob.glob(os.path.join(folder, "*.mp4")) + glob.glob(os.path.join(folder, "*.avi"))
    for v in videos:
        # แทนที่ backslash เป็น slash เพื่อป้องกัน error ในบางระบบ
        clean_path = v.replace("\\", "/")
        all_videos.append(f"{clean_path} {label}\n")

# 3. สลับข้อมูล (Shuffle) และแบ่งสัดส่วน Train 80% / Val 20%
random.shuffle(all_videos)
split_idx = int(len(all_videos) * 0.7)

train_data = all_videos[:split_idx]
val_data = all_videos[split_idx:]

# 4. เขียนลงไฟล์
with open("dataset/train.csv", "w", encoding="utf-8") as f:
    f.writelines(train_data)
with open("dataset/val.csv", "w", encoding="utf-8") as f:
    f.writelines(val_data)

print(f"สร้าง CSV สำเร็จ! Train: {len(train_data)} คลิป, Val: {len(val_data)} คลิป")
