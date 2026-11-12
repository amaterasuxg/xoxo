import glob

# เปลี่ยนพาทให้ตรงกับโฟลเดอร์ของคุณ (เช่น ucf_dataset/valid/labels/*.txt)
label_files = glob.glob("ucf_dataset/valid/labels/*.txt") 

for file_path in label_files:
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # บังคับเปลี่ยนตัวเลขหน้าสุดเป็น 0 (Person)
            new_line = f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
            new_lines.append(new_line)
            
    with open(file_path, "w") as f:
        f.writelines(new_lines)

print(f"แปลงไฟล์เสร็จสิ้น จำนวน {len(label_files)} ไฟล์!")