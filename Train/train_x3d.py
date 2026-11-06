# ==========================================
# 0. โค้ดแก้ปัญหาไลบรารี pytorchvideo
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
import matplotlib.pyplot as plt

# ==========================================
# 1. 🛠️ HYPER-PARAMETERS
# ==========================================
NUM_CLASSES = 3
BATCH_SIZE = 24
EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "gpu")

# --- เปิดการเชื่อมต่อ WandB (ชื่อรัน X3D_3Classes) ---
wandb.init(
    project="action-recognition-project",
    name="X3D_3Classes_100", 
    config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "model_architecture": "X3D_s",
        "classes": NUM_CLASSES
    }
)

# ==========================================
# 2. คลาสโหลดวิดีโอ (ส่งข้อมูลก้อนเดียวสำหรับ X3D)
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
        # Tensor Shape: (C, T, H, W)
        clip_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
        
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        clip_tensor = (clip_tensor - mean) / std
        
        # คืนค่าคลิปก้อนเดียว (ไม่ต้องแบ่ง Slow/Fast)
        return clip_tensor, label

# ==========================================
# 3. โหลดข้อมูล
# ==========================================
print("กำลังโหลดชุดข้อมูลสำหรับ X3D...")
train_dataset = OpenCVVideoDataset(csv_file="dataset/train.csv", image_size=IMAGE_SIZE)
val_dataset = OpenCVVideoDataset(csv_file="dataset/val.csv", image_size=IMAGE_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

# ==========================================
# 4. โหลดและเตรียมโมเดล AI (X3D)
# ==========================================
print("กำลังโหลดโมเดล X3D (x3d_s)...")
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)

# แช่แข็งพารามิเตอร์ทั้งหมด
for param in model.parameters():
    param.requires_grad = False

# ปลดล็อกสมองส่วนลึก (Block 5)
for param in model.blocks[5].parameters():
    param.requires_grad = True

# ปรับ Layer สุดท้ายให้เป็น 3 คลาส
in_features = model.blocks[5].proj.in_features
model.blocks[5].proj = nn.Linear(in_features, NUM_CLASSES)
for param in model.blocks[5].proj.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# ==========================================
# 5. ลูปการเทรน
# ==========================================
print(f"เริ่มการเทรนบน {DEVICE} (ทั้งหมด {EPOCHS} รอบ)...")
history_train_loss, history_val_loss = [], []
history_train_acc, history_val_acc = [], []

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    running_train_loss = 0.0
    correct_train, total_train = 0, 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(DEVICE)  # ส่งก้อนเดียวขึ้นการ์ดจอ
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
        vram_used = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Top-1 Acc: {running_acc:.2f}% | VRAM: {vram_used:.2f} GB")
    
    avg_train_loss = running_train_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # --- Validation ---
    model.eval()
    running_val_loss = 0.0
    correct_val, total_val = 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    avg_val_loss = running_val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val

    print(f"\n--- สรุป Epoch {epoch+1} (X3D) ---")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Top-1 Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {avg_val_loss:.4f} | Val Top-1 Acc:   {val_acc:.2f}%\n")
    
    history_train_loss.append(avg_train_loss)
    history_val_loss.append(avg_val_loss)
    history_train_acc.append(train_acc / 100)
    history_val_acc.append(val_acc / 100)
    
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": avg_train_loss,
        "Train Top-1 Accuracy": train_acc,
        "Validation Loss": avg_val_loss,
        "Validation Top-1 Accuracy": val_acc
    })

# ==========================================
# 6. บันทึกผลลัพธ์และวาดกราฟ
# ==========================================
torch.save(model.state_dict(), "custom_x3d_3class_run50.pth")
wandb.finish()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(history_train_loss, label='Train (X3D)', color='blue')
ax1.plot(history_val_loss, label='Validation (X3D)', color='red')
ax1.set_title('Loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend()

ax2.plot(history_train_acc, label='Train (X3D)', color='blue')
ax2.plot(history_val_acc, label='Validation (X3D)', color='red')
ax2.set_title('Top-1 Accuracy')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('train_val_graph_x3d.png', dpi=300)
print("🎉 เทรน X3D เสร็จสมบูรณ์! ได้ไฟล์ custom_x3d_3class.pth และ train_val_graph_x3d.png")