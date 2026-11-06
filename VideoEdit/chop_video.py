import cv2
import os
import uuid

# ==========================================
# 1. ตั้งค่าการหั่นคลิป (ปรับแก้ตรงนี้ได้เลย)
# ==========================================
VIDEO_PATH = "raw_video/fall.mp4"  # ชื่อไฟล์วิดีโอต้นฉบับที่ถ่ายมายาวๆ
OUTPUT_FOLDER = "dataset/videos"              # โฟลเดอร์ที่จะเก็บคลิปที่หั่นแล้ว
CLASS_NAME = "fall"                          # ชื่อท่าทาง (เช่น work, not_work, fall)
FRAMES_PER_CLIP = 32                          # จำนวนเฟรมต่อ 1 คลิป (SlowFast นิยมใช้ 32 หรือ 64 เฟรม)

# ==========================================
# 2. ฟังก์ชันหั่นวิดีโอ
# ==========================================
def slice_video(video_path, output_dir, class_name, frames_per_clip):
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ไม่สามารถเปิดไฟล์วิดีโอได้: {video_path}")
        return

    # ดึงค่า FPS และขนาดภาพจากต้นฉบับ
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # กันเหนียวถ้าอ่านค่า fps ไม่ได้
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ตั้งค่า VideoWriter สำหรับ MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    frame_buffer = []
    clip_count = 0
    
    print(f"🎬 กำลังหั่นวิดีโอ {video_path} ...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # จบวิดีโอ
            
        frame_buffer.append(frame)
        
        # เมื่อสะสมเฟรมครบตามที่กำหนด (เช่น 32 เฟรม) ให้เซฟเป็น 1 คลิป
        if len(frame_buffer) == frames_per_clip:
            # 1. สร้างชื่อไฟล์ไม่ให้ซ้ำ (ใช้ class_name + รหัสสุ่ม 8 ตัวอักษร)
            unique_id = uuid.uuid4().hex[:8]
            filename = f"{class_name}_{unique_id}.mp4"
            filepath = os.path.join(output_dir, filename)
            
            # 2. บันทึกวิดีโอ
            out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
            for f in frame_buffer:
                out.write(f)
            out.release()
            
            clip_count += 1
            print(f"✅ สร้างคลิปที่ {clip_count}: {filename}")
            
            # 3. ล้างตะกร้า เพื่อเริ่มเก็บเฟรมของคลิปถัดไป
            frame_buffer = []

    cap.release()
    print(f"\n🎉 หั่นเสร็จสมบูรณ์! ได้คลิปท่า {class_name} ทั้งหมด {clip_count} คลิป")
    print(f"📂 ไฟล์ทั้งหมดถูกเก็บไว้ที่: {output_dir}")

# ==========================================
# 3. สั่งเริ่มทำงาน
# ==========================================
if __name__ == "__main__":
    slice_video(VIDEO_PATH, OUTPUT_FOLDER, CLASS_NAME, FRAMES_PER_CLIP)