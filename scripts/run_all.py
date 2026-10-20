"""对 7 个序列批量跑自己写的跟踪器 v3 和 v4，结果写成 MOT 格式。

用法（在项目根目录 mot17-tracking/ 下运行）：
    python tools/run_all.py
输出：results/v3/data/{seq}.txt 和 results/v4/data/{seq}.txt
"""
import os
import sys
import time

# 把项目根目录加进搜索路径，这样才能 import 根目录下的 tracker / mot_io / box_utils
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from mot_io import load_dets
from tracker import track_step_v3, track_step_v4
from box_utils import xyxy_to_xywh

SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]


def run_seq(seq, version):
    """和 run_track.py 一样的循环；version 是 "v3" 或 "v4" """
    dets = load_dets(f"dets/{seq}.txt")
    n_frames = len(os.listdir(f"data/MOT17/train/{seq}/img1"))
    out_dir = f"results/{version}/data"
    os.makedirs(out_dir, exist_ok=True)

    tracks = {}
    missed = {}
    vels = {}
    next_id = 1
    with open(f"{out_dir}/{seq}.txt", "w") as f:
        for frame in range(1, n_frames + 1):
            frame_dets = dets.get(frame, [])
            if version == "v3":
                # v3 只吃高分框（分数 >= 0.5）
                high = [(box, score) for box, score in frame_dets if score >= 0.5]
                tracks, missed, vels, next_id = track_step_v3(tracks, missed, vels, high, next_id)
            else:
                # v4 吃全部框，内部自己分高低分
                tracks, missed, vels, next_id = track_step_v4(tracks, missed, vels, frame_dets, next_id)
            # 只写这一帧真的匹配上的轨迹（missed == 0）
            for tid, box in tracks.items():
                if missed[tid] == 0:
                    x, y, w, h = xyxy_to_xywh(box)
                    f.write(f"{frame},{tid},{x:.1f},{y:.1f},{w:.1f},{h:.1f},1,-1,-1,-1\n")
    return next_id - 1


if __name__ == "__main__":
    for version in ["v3", "v4"]:
        t0 = time.time()
        for seq in SEQS:
            n = run_seq(seq, version)
            print(version, seq, "一共发了", n, "个编号")
        print(version, "用时 %.1f 秒" % (time.time() - t0))
