def predict(box, vel):
    """按速度把框往前推一帧"""
    result = []
    for b, v in zip(box, vel):
        result.append(b+v)

    return tuple(result)


def update(pred, meas, vel, alpha=0.75, beta=0.1):
    """用检测框修正预测：返回 (修正后的框, 修正后的速度)"""
    new_box = []
    new_vel = []

    for p, m, v in zip(pred, meas, vel):
        diff = m - p
        new_box.append(p + alpha*diff)
        new_vel.append(v + beta*diff)

    return tuple(new_box),tuple(new_vel)
