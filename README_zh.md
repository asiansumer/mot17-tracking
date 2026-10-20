# MOT17 行人多目标跟踪：从 IoU 跟踪器一步步写到 ByteTrack

[English](README.md)

这个项目用我上一个项目 [yolov8-bdd100k-pedestrian](https://github.com/asiansumer/yolov8-bdd100k-pedestrian) 里训练的 YOLOv8m 行人检测模型，在 MOT17 的 7 个训练序列上做多目标跟踪。跟踪器是我自己从零写的，前后改了四个版本，从最简单的 IoU 配对一直到 ByteTrack 式的两轮配对。最后我用 Ultralytics 自带的 ByteTrack 作参照，三个跟踪器统一用 TrackEval 评测。

结果是：我自己写的 v4 和 Ultralytics 的 ByteTrack 差不多（HOTA 40.6 对 41.2），两边的漏检数几乎一样，所以现在拖后腿的主要是检测器，不是跟踪器。另外在 MOT17-05 这个序列上，加入低分框以后 MOTA 反而从 38.9 掉到了 21.7，我花了不少时间才弄清楚原因（见[发现 2](#发现-2低分框在-05-上反而有害)）。

## 为什么自己写

调用现成的跟踪器只要一行代码，但我只能拿到结果，不知道它每一步在做什么，也说不清它为什么会出错。所以我从最简单的版本写起，每次只解决上一版暴露的一个问题。

## 四个版本

| 版本 | 做了什么 | 解决什么问题 |
|---|---|---|
| v1 | 每一帧用 IoU 把新检测框和上一帧的人配对，配不上的发新编号 | 最基础的跟踪 |
| v2 | 暂时消失的人先放进“等待区”，30 帧内再出现就沿用原来的编号 | 人被短暂挡住后编号就变 |
| v3 | 加入 alpha-beta 速度预测，用预测出来的位置去配对 | 人在走，上一帧的位置和这一帧对不上；被挡住时也能按速度往前推 |
| v4 | ByteTrack 式两轮配对：先用高分框配，剩下没配上的老轨迹再去低分框里找 | v3 只用分数 ≥ 0.5 的框，被挡住一半的人分数常常掉到 0.5 以下，直接被扔掉 |

v4 的关键规则是：**低分框只能续接已有的轨迹，不能开新编号。** 人被挡住时，检测器对他没那么确定，框还在，只是分数变低了，所以低分框里有不少是“老熟人”。但低分框里也有很多误检，如果让它们开新编号，就会冒出一堆不存在的人。

这样做的代价是误检变多：从 v3 到 v4，漏检（FN）少了约 12000 个，误检（FP）多了约 8000 个。整体上 MOTA 提高了 3.6，IDF1 提高了 4.7。

## 结果

7 个序列合计，用 TrackEval 评测（MOT17 train，FRCNN 那一版的 7 个序列）：

| 跟踪器 | HOTA | MOTA | IDF1 | FN | FP | IDSW | 编号数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3（只用分数 ≥ 0.5 的框） | 36.9 | 38.9 | 43.8 | 63118 | 4577 | 957 | 1211 |
| v4（自己写的 ByteTrack） | 40.6 | 42.5 | 48.5 | 50974 | 12643 | 964 | 937 |
| Ultralytics ByteTrack | 41.2 | 43.2 | 50.1 | 51103 | 11681 | 1031 | 1383 |

真实人数（GT_IDs）是 546。

每个序列的 MOTA / IDF1：

| 序列 | v4 MOTA | v4 IDF1 | Ultralytics MOTA | Ultralytics IDF1 |
|---|---:|---:|---:|---:|
| 02 | 26.4 | 36.7 | 30.0 | 37.9 |
| 04 | 47.5 | 50.7 | 46.4 | 53.1 |
| 05 | 21.7 | 49.0 | 26.0 | 49.3 |
| 09 | 59.6 | 43.9 | 62.0 | 56.1 |
| 10 | 50.8 | 51.6 | 50.9 | 44.7 |
| 11 | 45.6 | 53.1 | 46.7 | 61.0 |
| 13 | 40.4 | 50.9 | 41.4 | 49.8 |

需要说明的是，这不是完全公平的对比：v4 读的是我提前跑好的检测结果（`dets/`，conf ≥ 0.1），Ultralytics 是每一帧自己重新检测，用的是它默认的阈值。

### 发现 1：瓶颈在检测器，不在跟踪器

两个跟踪器的漏检都在 51000 左右，几乎一样。我的 v4 只比 Ultralytics 低 0.6 HOTA，发出的编号还少了 446 个，说明给同一个人换号更少。

我觉得原因在检测器：这个 YOLOv8m 是在 BDD100K 上训练的，那是车载摄像头的视角；MOT17 大部分是街边或手持拍的，视角、人的大小和遮挡方式都不一样。跟踪器再怎么改，检测器没框出来的人也补不回来。

### 发现 2：低分框在 05 上反而有害

在其他序列上 v4 都比 v3 好，只有 MOT17-05 反过来：MOTA 从 38.9 掉到 21.7，但 IDF1 基本没变（v4 是 49.0）。

05 是跟在一个行人后面边走边拍的，前面的人经常挡住后面的人。我一开始以为多出来的误检是远处的小人或者车，但把 v4 输出的框和真实框一个个比之后，发现多出来的框大部分不是“空的”，而是套在一个真人身上但位置歪了（IoU 在 0.2 到 0.5 之间）：

- 这些歪框里 89% 比对应的真人框矮，高度的中位数只有真人框的 0.67；
- 对应的真人平均只有 46% 的身体露在外面。

原因是 MOT17 的真实框画的是整个人，被挡住的部分也画进去；而检测器只能框到看得见的那一半。半身框和全身框的 IoU 很容易低于 0.5，评测时会被罚两次：它自己算一个误检，旁边那个真人又算一个漏检。v4 专门捡低分框来续轨迹，捡回来的正好很多是这种半身框。

这也解释了为什么 IDF1 没跟着掉：编号其实更稳了（换号 241 → 223 次），问题出在框的位置上。

## 局限和下一步

- 检测器是车载视角训练的，没在 MOT17 这种街拍数据上训练过，这是现在最大的瓶颈。
- 配对用的是贪心法（轨迹按顺序挑 IoU 最大的框），不是全局最优的匈牙利算法。
- 速度预测用的是简单的 alpha-beta，不是卡尔曼滤波。
- 只看框的位置，不看人长什么样，两个人交叉走过时还是容易换号（7 个序列一共 964 次）。
- 下一步我想把它放到车载视角的跟踪数据上跑，比如 BDD100K 的 MOT 部分，这样和我上一个项目以及自动驾驶的场景都能接上。

## 怎么运行

数据和模型不在仓库里：MOT17 从 [motchallenge.net](https://motchallenge.net/data/MOT17/) 下载，解压到 `data/MOT17/`；模型权重从[上一个项目的 release](https://github.com/asiansumer/yolov8-bdd100k-pedestrian/releases) 下载 `yolov8m_bdd100k_pedestrian.pt`，放到项目根目录并改名为 `best.pt`。评测用的 [TrackEval](https://github.com/JonathonLuiten/TrackEval) 放在和本项目同级的目录。

以下命令都在项目根目录下运行：

```bash
# 1. 用 YOLOv8m 检测 7 个序列，结果写到 dets/
python -m scripts.detect_seq

# 2. 跑自己写的 v3 和 v4，结果写到 results/v3、results/v4
python -m scripts.run_all

# 3. 跑 Ultralytics 的 ByteTrack 作参照，结果写到 results/ul_bytetrack
python -m scripts.run_ultralytics_bytetrack

# 4. 用 TrackEval 评测三个跟踪器
python ../TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train \
  --GT_FOLDER data/MOT17/train --TRACKERS_FOLDER results --SKIP_SPLIT_FOL True \
  --TRACKERS_TO_EVAL v3 v4 ul_bytetrack \
  --SEQ_INFO MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN \
  --METRICS HOTA CLEAR Identity --USE_PARALLEL False --PLOT_CURVES False
```

## 仓库结构

```
tracker.py      跟踪器 v1–v4
motion.py       alpha-beta 速度预测
box_utils.py    IoU、框格式转换
mot_io.py       读取 GT 和检测结果
scripts/        检测、跟踪、对照实验、画图
practice/       练习用的小脚本
```
