import os
from mot_io import load_dets
from tracker import track_step_v2
from box_utils import xyxy_to_xywh


def run_seq(seq, score_thr=0.5):
    dets = load_dets(f"dets/{seq}.txt")
    n_frames = len(os.listdir(f"data/MOT17/train/{seq}/img1"))
    os.makedirs("results", exist_ok=True)

    tracks = {}
    missed = {}    # 一开始档案是空的
    next_id = 1       # 取号机从 1 开始
    with open(f"results/{seq}.txt", "w") as f:
        for frame in range(1, n_frames + 1):
            # 1. 这一帧只留高分框（第 3 关先不用低分框）
            high = []
            for box, score in dets.get(frame, []):
                if score >= score_thr:
                    high.append((box, score))
            # 2. 跟踪一帧
            tracks, missed, next_id = track_step_v2(tracks, missed, high, next_id)
            # 3. 把这一帧档案里的每个人写进结果文件（MOT 格式）
            for tid, box in tracks.items():
                if missed[tid] == 0:
                    x, y, w, h = xyxy_to_xywh(box)
                    f.write(f"{frame},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,-1,-1,-1\n")

    print(seq, "一共发了", next_id - 1, "个编号")


run_seq("MOT17-02-FRCNN")
