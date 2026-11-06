# ==========================================
# 0. โค้ดแก้ปัญหาไลบรารี pytorchvideo
# ==========================================
import sys
import torchvision.transforms.functional as F_vision
sys.modules['torchvision.transforms.functional_tensor'] = F_vision

import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. ตั้งค่าพื้นฐาน
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
CLASSES = ["Working", "Not_Working", "Falling"] # ชื่อโฟลเดอร์ต้องตรงกับที่สกัดภาพไว้
FRAMES_PER_CLIP = 32 # อิงตามที่ test_pipeline.py ใช้ (เก็บ 32 เฟรม)
BATCH_SIZE = 4       # ปรับลดได้ถ้า VRAM ไม่พอ (เช่น 2)
EPOCHS = 20

# ==========================================
# 2. คลาส DataLoader สำหรับอ่านภาพที่ถูก Crop
# ==========================================
class CroppedVideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=32):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.data = []
        
        # เข้าไปสำรวจโฟลเดอร์ Working, Not_Working, Falling
        for class_idx, class_name in enumerate(CLASSES):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path): 
                continue
                
            # สำรวจแต่ละวิดีโอ (video_0000, video_0001...)
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                if os.path.isdir(video_path):
                    frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
                    
                    # คัดเฉพาะวิดีโอที่ตัดภาพคนมาได้อย่างน้อย 16 เฟรมขึ้นไป
                    if len(frames) >= 16: 
                        self.data.append((frames, class_idx))
                        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        frames_paths, label = self.data[idx]
        
        # ดึงรูปภาพให้กระจายทั่วทั้งคลิปให้ครบ 32 เฟรม
        total_frames = len(frames_paths)
        indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int)
        
        frame_buffer = []
        for i in indices:
            img = cv2.imread(frames_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) # รีไซส์ตามที่โมเดลต้องการ
            frame_buffer.append(img)
            
        video_np = np.array(frame_buffer)
        
        # แปลงเป็น Tensor (C, T, H, W) และ Normalize สี
        clip_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        clip_tensor = (clip_tensor - mean) / std
        
        # แยกเป็นเส้นทาง Slow (8 เฟรม) และ Fast (32 เฟรม) แบบเดียวกับตอน Test
        fast = clip_tensor
        slow_idx = torch.linspace(0, fast.shape[1] - 1, fast.shape[1] // 4).long()
        slow = torch.index_select(fast, 1, slow_idx)
        
        return [slow, fast], label

# ==========================================
# 3. โค้ดสำหรับเทรน (Training Loop)
# ==========================================
def main():
    print(f"กำลังรันบน: {DEVICE}")
    
    # โหลด Dataset
    train_dataset = CroppedVideoDataset("dataset/slowfast_data/train", clip_len=FRAMES_PER_CLIP)
    val_dataset = CroppedVideoDataset("dataset/slowfast_data/val", clip_len=FRAMES_PER_CLIP)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"จำนวนคลิป Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # ---------------------------------------------------------
    # โหลดโมเดล (ใช้ pretrained=True เพื่อเรียนรู้ต่อยอด จะแม่นขึ้นไวมาก)
    # ---------------------------------------------------------
    print("กำลังโหลดโมเดล SlowFast...")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    
    # เปลี่ยนหัวโมเดลท้ายสุดให้จำคลาสเรา 3 คลาส
    in_features = model.blocks[6].proj.in_features
    model.blocks[6].proj = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # ใช้ Learning Rate ต่ำๆ เพราะเป็น Fine-tune
    
    best_val_acc = 0.0
    
    print("\n🚀 เริ่มต้นการเทรน...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = [i.to(DEVICE) for i in inputs]
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        # ตรวจสอบความแม่นยำด้วย Val Dataset
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = [i.to(DEVICE) for i in inputs]
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = 100 * correct_val / total_val if total_val > 0 else 0
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # เซฟโมเดลตัวที่ดีที่สุดไว้
        if val_acc >= best_val_acc and val_acc > 0:
            best_val_acc = val_acc
            save_name = "custom_slowfast_3class_best.pth"
            torch.save(model.state_dict(), save_name)
            print(f"⭐ อัปเดตโมเดลที่ดีที่สุดเซฟเป็น '{save_name}' (Acc: {best_val_acc:.2f}%)")

if __name__ == '__main__':
    main()