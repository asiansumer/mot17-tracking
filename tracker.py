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

def track_step_v2(tracks, missed, dets, next_id, iou_thr=0.3, max_age=30):
    ''' v2加入missed，用于把暂时消失（被挡住）的人放进“等待区”在一定帧数内再次出现就沿用原来的编号'''
    new_tracks = {}
    new_missed = {}
    used = [False] * len(dets)          # 每个检测框有没有被认领

    for tid, tbox in tracks.items():
        best_j, best_v = -1, iou_thr
        for j, (box, score) in enumerate(dets):
            if used[j]:
                continue
            v = iou(box, tbox)
            if v > best_v:
                best_j, best_v = j, v
        if best_j >= 0:
            used[best_j] = True
            new_tracks[tid] = dets[best_j][0]
            new_missed[tid] = 0
        else:
            new_missed[tid] = missed[tid] + 1
            if new_missed[tid] <= max_age:
                new_tracks[tid] = tbox
    for j, (box, score) in enumerate(dets):
        if not used[j]:
            new_tracks[next_id] = box
            new_missed[next_id] = 0
            next_id += 1
    return new_tracks,new_missed, next_id
