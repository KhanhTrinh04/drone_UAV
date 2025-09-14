import os, random, shutil

# Gốc folder chứa ảnh + nhãn
src_dir = "./datasets/dataset_txt"

# Tạo folder theo cấu trúc YOLO
dst_img_train = "./datasets/dataset_txt/images/train"
dst_img_val   = "./datasets/dataset_txt/images/val"
dst_lbl_train = "./datasets/dataset_txt/labels/train"
dst_lbl_val   = "./datasets/dataset_txt/labels/val"

for d in [dst_img_train, dst_img_val, dst_lbl_train, dst_lbl_val]:
    os.makedirs(d, exist_ok=True)

# Lấy danh sách ảnh
images = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]
random.shuffle(images)

# Chia 80% train, 20% val
split_idx = int(0.8 * len(images))
train_files = images[:split_idx]
val_files = images[split_idx:]

def move_files(files, img_dst, lbl_dst):
    for f in files:
        img_src = os.path.join(src_dir, f)
        lbl_src = os.path.join(src_dir, f.replace(".jpg", ".txt"))
        if os.path.exists(lbl_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)

# Copy train và val
move_files(train_files, dst_img_train, dst_lbl_train)
move_files(val_files, dst_img_val, dst_lbl_val)

print("Done. Train size:", len(train_files), "Val size:", len(val_files))
