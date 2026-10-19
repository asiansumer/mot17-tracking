from tracker import track_step

# 3 帧的检测结果（编的小例子）
frames = {
    1: [((0, 0, 10, 10), 0.9), ((20, 0, 30, 10), 0.9)],
    2: [((1, 0, 11, 10), 0.9), ((50, 0, 60, 10), 0.8)],
    3: [((2, 0, 12, 10), 0.9), ((51, 0, 61, 10), 0.9), ((80, 0, 90, 10), 0.9)],
}

tracks = {}       # 一开始档案是空的
next_id = 1       # 取号机从 1 开始
for frame in [1, 2, 3]:
    tracks, next_id = track_step(tracks, frames[frame], next_id)
    print("第", frame, "帧的档案：", tracks)
