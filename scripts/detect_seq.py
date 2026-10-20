import os
from ultralytics import YOLO

model = YOLO("best.pt")
def detect_seq(seq):
    img_dir = f"data/MOT17/train/{seq}/img1"
    n_frames = len(os.listdir(img_dir))   # 文件夹里有几张图，就有几帧
    os.makedirs("dets", exist_ok=True)     # 新建 dets 文件夹放结果，已存在也不报错
    out_path = f"dets/{seq}.txt"

    with open(out_path, "w") as f:         # "w" 是写入模式：新建这个文件（已有就清空重写）
        for frame in range(1, n_frames+1):
            img_path = f"{img_dir}/{frame:06d}.jpg"
            results = model(img_path, conf=0.1, device="mps", verbose=False)
            boxes = results[0].boxes
            for box, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                f.write(f"{frame},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{score:.3f}\n")
            if frame % 100 == 0:
                print("已完成", frame, "帧")

SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]

for seq in SEQS:
    print("开始", seq)
    detect_seq(seq)
