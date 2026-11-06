import sys
import torchvision.transforms.functional as F_vision
sys.modules['torchvision.transforms.functional_tensor'] = F_vision

import os
import glob
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from fvcore.nn import FlopCountAnalysis

# ==========================================
# 1. ตั้งค่าพื้นฐาน
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["Working", "Not_Working", "Falling"]

# ตั้งชื่อไฟล์ .pth ที่คุณเทรนเสร็จแล้ว (แก้ชื่อให้ตรงกับของคุณ)
WEIGHTS = {
    "SlowFast": "custom_slowfast_3class_best.pth",
    "X3D_S": "custom_x3d_3class_run50.pth",         # ถ้ายังไม่มีไฟล์นี้ โค้ดจะข้ามไปอัตโนมัติ
    "ResNet3D": "custom_resnet_3class_run50.pth"    # ถ้ายังไม่มีไฟล์นี้ โค้ดจะข้ามไปอัตโนมัติ
}

# ==========================================
# 2. คลาสโหลดข้อมูล (เหมือนตอนเทรน)
# ==========================================
class CroppedVideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=32, model_type="slowfast"):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.model_type = model_type
        self.data = []
        for class_idx, class_name in enumerate(CLASSES):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path): continue
            for video_folder in os.listdir(class_path):
                video_path = os.path.join(class_path, video_folder)
                if os.path.isdir(video_path):
                    frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
                    if len(frames) >= 16: self.data.append((frames, class_idx))
                        
    def __len__(self): return len(self.data)
        
    def __getitem__(self, idx):
        frames_paths, label = self.data[idx]
        indices = np.linspace(0, len(frames_paths) - 1, self.clip_len).astype(int)
        
        frame_buffer = []
        for i in indices:
            img = cv2.imread(frames_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # X3D ใช้ภาพ 160x160 ส่วนตัวอื่นใช้ 224x224
            img_size = (160, 160) if self.model_type == "x3d" else (224, 224)
            img = cv2.resize(img, img_size)
            frame_buffer.append(img)
            
        video_np = np.array(frame_buffer)
        clip_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
        mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
        clip_tensor = (clip_tensor - mean) / std
        
        if self.model_type == "slowfast":
            fast = clip_tensor
            slow_idx = torch.linspace(0, fast.shape[1] - 1, fast.shape[1] // 4).long()
            slow = torch.index_select(fast, 1, slow_idx)
            return [slow, fast], label
        else:
            return clip_tensor, label

# ==========================================
# 3. โหลดและประเมินผลโมเดล
# ==========================================
def evaluate_model(name, model_func, weight_path, model_type, clip_len):
    if not os.path.exists(weight_path):
        print(f"⚠️ ข้าม {name}: ไม่พบไฟล์น้ำหนัก '{weight_path}'")
        return None

    print(f"\n🚀 กำลังโหลดและประเมินผล {name}...")
    model = model_func().to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    # 1. วัด Params & GFLOPs
    params = sum(p.numel() for p in model.parameters()) / 1e6
    dummy_input = torch.randn(1, 3, clip_len, 160, 160).to(DEVICE) if model_type == "x3d" else torch.randn(1, 3, clip_len, 224, 224).to(DEVICE)
    
    if model_type == "slowfast":
        slow_idx = torch.linspace(0, clip_len - 1, clip_len // 4).long()
        slow_dummy = torch.index_select(dummy_input, 2, slow_idx.to(DEVICE))
        dummy_input = ( [slow_dummy, dummy_input], )
    else:
        dummy_input = (dummy_input,)
        
    flops = FlopCountAnalysis(model, dummy_input).total() / 1e9

    # 2. วัด Top-1 Accuracy ด้วยข้อมูล Val
    val_dataset = CroppedVideoDataset("dataset/slowfast_data/val", clip_len=clip_len, model_type=model_type)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            if model_type == "slowfast":
                inputs = [i.to(DEVICE) for i in inputs]
            else:
                inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    top1_acc = (correct / total) * 100 if total > 0 else 0
    return {"Model": name, "Top-1 Acc": top1_acc, "GFLOPs": flops, "Params (M)": params}

# ฟังก์ชันเตรียมสถาปัตยกรรมโมเดล
def get_slowfast():
    m = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=False)
    m.blocks[6].proj = nn.Linear(m.blocks[6].proj.in_features, 3)
    return m

def get_x3d():
    m = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=False)
    m.blocks[5].proj = nn.Linear(m.blocks[5].proj.in_features, 3)
    return m

def get_resnet3d():
    m = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
    m.blocks[5].proj = nn.Linear(m.blocks[5].proj.in_features, 3)
    return m

def main():
    results = []
    
    # รันประเมินทีละตัว (ตั้งค่าเฟรมให้ตรงกับตอนเทรน)
    res_sf = evaluate_model("SlowFast", get_slowfast, WEIGHTS["SlowFast"], "slowfast", 32)
    if res_sf: results.append(res_sf)
        
    res_x3d = evaluate_model("X3D_S", get_x3d, WEIGHTS["X3D_S"], "x3d", 13)
    if res_x3d: results.append(res_x3d)
        
    res_res = evaluate_model("ResNet3D", get_resnet3d, WEIGHTS["ResNet3D"], "resnet", 8)
    if res_res: results.append(res_res)

    # ปริ้นท์ตารางผลลัพธ์
    print("\n" + "="*55)
    print(f"| {'Model':<12} | {'Top-1 Acc (%)':<13} | {'GFLOPs':<8} | {'Params (M)':<10} |")
    print("="*55)
    for r in results:
        print(f"| {r['Model']:<12} | {r['Top-1 Acc']:<13.2f} | {r['GFLOPs']:<8.2f} | {r['Params (M)']:<10.2f} |")
    print("="*55)

if __name__ == '__main__':
    main()