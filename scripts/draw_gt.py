import cv2
from mot_io import load_gt

seq = "data/MOT17/train/MOT17-02-FRCNN"
frames = load_gt(seq + "/gt/gt.txt")

frame = 1
img = cv2.imread(f"{seq}/img1/{frame:06d}.jpg")   # 读入第 1 帧的图片
if img is None:
    raise FileNotFoundError("找不到图片，检查路径和当前文件夹")
for pid, box in frames[frame]:
    x1, y1, x2, y2 = box
    # 画绿色的框，线粗 2
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # 在框的左上角写上人的编号
    cv2.putText(img, str(pid), (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite("frame1_gt.jpg", img)   # 把画好的图存成文件
print("画了", len(frames[frame]), "个人")
