

def xywh_to_xyxy(box):
    """把 (起点x, 起点y, 宽, 高) 换成 (起点x, 起点y, 终点x, 终点y)"""
    x, y, w, h = box          # 把 4 个数一次拆开，分别放进 x、y、w、h
    return (x,y,x+w,y+h)
