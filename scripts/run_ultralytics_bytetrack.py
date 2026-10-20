"""用 ultralytics 自带的 ByteTrack 跑 7 个序列，作为和自己写的 v4 对照的基线。

用法（在项目根目录 mot17-tracking/ 下运行，用 bdd 环境的 python）：
    python tools/run_ultralytics_bytetrack.py
输出：results/ul_bytetrack/data/{seq}.txt（MOT 格式）
"""
import os
import time

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
OUT_DIR = "results/ul_bytetrack/data"


def run_seq(seq):
    # 每个序列新建一个 YOLO 对象，等于把跟踪器状态清零
    model = YOLO("best.pt")
    img_dir = f"data/MOT17/train/{seq}/img1"
    names = sorted(os.listdir(img_dir))      # 000001.jpg, 000002.jpg ... 按帧顺序
    with open(f"{OUT_DIR}/{seq}.txt", "w") as f:
        for i, name in enumerate(names):
            frame = i + 1
            # persist=True：让跟踪器记住上一帧的轨迹
            r = model.track(f"{img_dir}/{name}", persist=True, tracker="bytetrack.yaml",
                            device="mps", verbose=False)[0]
            if r.boxes.id is None:           # 这一帧没有带编号的框
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                f.write(f"{frame},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1\n")
    return len(names)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.time()
    for seq in SEQS:
        t0 = time.time()
        n = run_seq(seq)
        print(seq, n, "帧，用时 %.1f 秒" % (time.time() - t0))
    print("总用时 %.1f 秒" % (time.time() - t_all))
