# ==========================================
# 0. โค้ดแก้ปัญหาไลบรารี pytorchvideo (ต้องอยู่บนสุดเสมอ)
# ==========================================
import sys
import torchvision.transforms.functional as F_vision
sys.modules['torchvision.transforms.functional_tensor'] = F_vision

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
import matplotlib.pyplot as plt  # ไลบรารีสำหรับวาดกราฟ

# ==========================================
# 1. 🛠️ HYPER-PARAMETERS (ปรับเป็น 3 คลาสแล้ว)
# ==========================================
NUM_CLASSES = 3  # Working (0), Not Working (1), Falling (2)
BATCH_SIZE = 24
EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "gpu")

# --- เปิดการเชื่อมต่อ WandB ---
wandb.init(
    project="action-recognition-project",
    name="SlowFast_3Classes_Run100", # ตั้งชื่อ Run
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "model_architecture": "SlowFast",
        "classes": NUM_CLASSES
    }
)

# ==========================================
# 2. คลาสโหลดวิดีโอ (Custom OpenCV)
# ==========================================
class OpenCVVideoDataset(Dataset):
    def __init__(self, csv_file, num_frames=32, image_size=224):
        self.data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    path, label = line.strip().rsplit(' ', 1)
                    self.data.append((path, int(label)))
                    
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if cap.isOpened():
            total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            current_frame, frame_idx = 0, 0
            while cap.isOpened() and len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret: break
                
                if current_frame == indices[frame_idx]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                    frames.append(frame)
                    frame_idx += 1
                    while frame_idx < self.num_frames and current_frame == indices[frame_idx]:
                        frames.append(frame)
                        frame_idx += 1
                current_frame += 1
            cap.release()

        while len(frames) < self.num_frames:
            frames.append(frames[-1] if len(frames) > 0 else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        video_np = np.array(frames)
        clip_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
        
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        clip_tensor = (clip_tensor - mean) / std
        
        fast = clip_tensor
        slow_idx = torch.linspace(0, fast.shape[1] - 1, fast.shape[1] // 4).long()
        slow = torch.index_select(fast, 1, slow_idx)
        
        return [slow, fast], label

# ==========================================
# 3. โหลดข้อมูล Train และ Val
# ==========================================
print("กำลังโหลดชุดข้อมูลจาก dataset ด้วย OpenCV...")
train_dataset = OpenCVVideoDataset(csv_file="dataset/train.csv", image_size=IMAGE_SIZE)
val_dataset = OpenCVVideoDataset(csv_file="dataset/val.csv", image_size=IMAGE_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

# ==========================================
# 4. โหลดและเตรียมโมเดล AI (SlowFast)
# ==========================================
print("กำลังโหลดโมเดล SlowFast ต้นฉบับ...")
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

# แช่แข็งพารามิเตอร์ทั้งหมด
for param in model.parameters():
    param.requires_grad = False

# ปลดล็อก Block 5 
for param in model.blocks[5].parameters():
    param.requires_grad = True

# ปรับ Layer สุดท้ายให้เป็น 3 คลาส และปลดล็อก
in_features = model.blocks[6].proj.in_features
model.blocks[6].proj = nn.Linear(in_features, NUM_CLASSES)
for param in model.blocks[6].proj.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# ==========================================
# 5. ลูปการเทรน
# ==========================================
print(f"เริ่มการเทรนบน {DEVICE} (ทั้งหมด {EPOCHS} รอบ)...")

# ตัวแปรสำหรับเก็บประวัติไปวาดกราฟ
history_train_loss = []
history_val_loss = []
history_train_acc = []
history_val_acc = []

for epoch in range(EPOCHS):
    # ---------------- ช่วง Train ----------------
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = [x.to(DEVICE) for x in inputs]
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        running_acc = 100 * correct_train / total_train
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Top-1 Acc: {running_acc:.2f}%")
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # ---------------- ช่วง Validation ----------------
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = [x.to(DEVICE) for x in inputs]
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    avg_val_loss = running_val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val

    # สรุปผลตอนจบ 1 Epoch
    print(f"\n--- สรุป Epoch {epoch+1} ---")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Top-1 Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {avg_val_loss:.4f} | Val Top-1 Acc:   {val_acc:.2f}%\n")
    
    # เก็บค่าลง List เพื่อเตรียมวาดกราฟ (แปลง Acc เป็น 0-1)
    history_train_loss.append(avg_train_loss)
    history_val_loss.append(avg_val_loss)
    history_train_acc.append(train_acc / 100)
    history_val_acc.append(val_acc / 100)
    
    # ส่งขึ้น WandB
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": avg_train_loss,
        "Train Top-1 Accuracy": train_acc,
        "Validation Loss": avg_val_loss,
        "Validation Top-1 Accuracy": val_acc
    })
        
# ==========================================
# 6. บันทึกผลลัพธ์
# ==========================================
torch.save(model.state_dict(), "custom_slowfast_3class_run50.pth")
wandb.finish()

# ==========================================
# 7. วาดกราฟ Train/Val แบบซ้อนกัน
# ==========================================
print("กำลังวาดกราฟและบันทึกรูปภาพ...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# (a) กราฟ Loss
ax1.plot(history_train_loss, label='Train', color='blue')
ax1.plot(history_val_loss, label='Validation', color='red')
ax1.set_title('Loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend()

# (b) กราฟ Top-1 Accuracy
ax2.plot(history_train_acc, label='Train', color='blue')
ax2.plot(history_val_acc, label='Validation', color='red')
ax2.set_title('Top-1 Accuracy')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('train_val_graph_3class.png', dpi=300)
print("🎉 เทรนเสร็จสมบูรณ์! ได้โมเดล custom_slowfast_3class.pth")
print("📸 บันทึกภาพกราฟ Train vs Val สำเร็จ! (ไฟล์: train_val_graph_3class.png)")