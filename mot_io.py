from box_utils import xywh_to_xyxy




def parse_gt_line(line):
    """把 gt.txt 的一行变成 (帧号, 人的编号, 框)；不留的行返回 None"""
    parts = line.strip().split(",")
    # 第 7 列不是 1，或者第 8 列不是 1，就不留
    if parts[6] != "1" or parts[7] != "1":
        return None
    frame = int(parts[0])
    pid = int(parts[1])
    box = (float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
    return (frame, pid, xywh_to_xyxy(box))


def load_gt(path):
    """读整个 gt.txt，返回 {帧号: [(人的编号, 框), ...]}"""
    frames = {}
    with open(path) as f:
        for line in f:
            result = parse_gt_line(line)
            if result is None:
                continue
            frame, pid, box = result
            if frame not in frames:
                frames[frame] = []
            frames[frame].append((pid, box))
    return frames
