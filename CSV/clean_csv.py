import os

def clean_csv(csv_path):
    # อ่านข้อมูลเดิม
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    valid_lines = []
    missing_count = 0

    # เช็คว่าไฟล์วิดีโอยังอยู่ไหม
    for line in lines:
        if not line.strip():
            continue
            
        video_path, label = line.strip().rsplit(' ', 1)
        
        # ถ้าไฟล์วิดีโอยังอยู่ในเครื่อง ให้เก็บบรรทัดนี้ไว้
        if os.path.exists(video_path):
            valid_lines.append(line)
        else:
            missing_count += 1
            print(f"ลบออกจาก CSV แล้ว: {video_path}")

    # เขียนทับไฟล์ CSV เดิมด้วยข้อมูลที่ถูกต้อง
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)

    print(f"\n✅ อัปเดต {csv_path} เสร็จสิ้น! (ลบข้อมูลที่ไม่มีไฟล์จริงออกไป {missing_count} รายการ)")

# เรียกใช้งานฟังก์ชันกับไฟล์ Train และ Val ของคุณ
# (แก้ชื่อไฟล์/path ให้ตรงกับของคุณถ้าจำเป็น)
clean_csv("dataset/train.csv")
clean_csv("dataset/val.csv")