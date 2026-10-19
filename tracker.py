from box_utils import iou


def track_step(tracks, dets, next_id, iou_thr=0.3):
    """处理一帧：返回 (新档案, 新的 next_id)"""
    new_tracks = {}
    used = [False] * len(dets)          # 每个检测框有没有被认领

    # 第一步：档案里的每个人，去找 IoU 最大、还没被认领的检测框
    for tid, tbox in tracks.items():
        best_j, best_v = -1, iou_thr
        for j, (box, score) in enumerate(dets):
            if used[j]:
                continue
            v = iou(box, tbox)
            if v > best_v:
                best_j, best_v = j, v
        if best_j >= 0:                  # 找到了：沿用编号，框更新成新位置
            used[best_j] = True
            new_tracks[tid] = dets[best_j][0]

    # 第二步：没被认领的检测框，发新号
    for j, (box, score) in enumerate(dets):
        if not used[j]:
            new_tracks[next_id] = box
            next_id += 1
    return new_tracks, next_id
