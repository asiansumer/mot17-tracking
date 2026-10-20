# Multi-Object Pedestrian Tracking on MOT17: from an IoU Tracker to ByteTrack

[中文版](README_zh.md)

This project takes the YOLOv8m pedestrian detector I trained in my previous project, [yolov8-bdd100k-pedestrian](https://github.com/asiansumer/yolov8-bdd100k-pedestrian), and uses it for multi-object tracking on the 7 MOT17 training sequences. I wrote the tracker myself from scratch and went through four versions, from plain IoU matching up to ByteTrack-style two-round matching. At the end I used Ultralytics' built-in ByteTrack as a reference, and evaluated all three trackers with TrackEval.

My v4 ended up close to Ultralytics' ByteTrack (HOTA 40.6 vs 41.2), and both miss almost the same number of people, so the detector is now the main thing holding the results back, not the tracker. On one sequence, MOT17-05, adding low-score boxes actually made MOTA drop from 38.9 to 21.7. It took me a while to work out why (see [Finding 2](#finding-2-low-score-boxes-hurt-on-mot17-05)).

## Why I wrote it myself

Calling an existing tracker takes one line of code, but then I only get the output. I don't know what it does at each step, and I can't explain why it fails when it does. So I started from the simplest version and only fixed one problem from the previous version each time.

## Four versions

| Version | What it does | Problem it fixes |
|---|---|---|
| v1 | Matches each new detection to last frame's people by IoU; unmatched detections get a new ID | Basic tracking |
| v2 | People who disappear go into a "waiting area"; if they come back within 30 frames they keep their old ID | IDs change after a short occlusion |
| v3 | Adds alpha-beta velocity prediction and matches against the predicted position | People move, so last frame's box doesn't line up with this frame; during occlusion the box keeps moving with the velocity |
| v4 | ByteTrack-style two rounds: match high-score boxes first, then try the leftover tracks against low-score boxes | v3 only uses boxes with score ≥ 0.5, and a half-occluded person often drops below 0.5 and gets thrown away |

The key rule in v4: **low-score boxes can only continue an existing track, never start a new ID.** When someone is partly hidden, the detector is less sure about them. The box is still there, just with a lower score, so many low-score boxes belong to people the tracker already knows. But low-score boxes also include a lot of false detections, and if they could start new IDs, a lot of people who don't exist would appear.

The cost is more false positives: from v3 to v4, missed detections (FN) went down by about 12,000 and false positives (FP) went up by about 8,000. Overall MOTA went up by 3.6 and IDF1 by 4.7.

## Results

All 7 sequences combined, evaluated with TrackEval (MOT17 train, the FRCNN versions of the 7 sequences):

| Tracker | HOTA | MOTA | IDF1 | FN | FP | IDSW | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 (only boxes with score ≥ 0.5) | 36.9 | 38.9 | 43.8 | 63118 | 4577 | 957 | 1211 |
| v4 (my ByteTrack) | 40.6 | 42.5 | 48.5 | 50974 | 12643 | 964 | 937 |
| Ultralytics ByteTrack | 41.2 | 43.2 | 50.1 | 51103 | 11681 | 1031 | 1383 |

There are 546 real people (GT_IDs) in total.

MOTA / IDF1 per sequence:

| Sequence | v4 MOTA | v4 IDF1 | Ultralytics MOTA | Ultralytics IDF1 |
|---|---:|---:|---:|---:|
| 02 | 26.4 | 36.7 | 30.0 | 37.9 |
| 04 | 47.5 | 50.7 | 46.4 | 53.1 |
| 05 | 21.7 | 49.0 | 26.0 | 49.3 |
| 09 | 59.6 | 43.9 | 62.0 | 56.1 |
| 10 | 50.8 | 51.6 | 50.9 | 44.7 |
| 11 | 45.6 | 53.1 | 46.7 | 61.0 |
| 13 | 40.4 | 50.9 | 41.4 | 49.8 |

This is not a completely fair comparison: v4 reads detections I ran in advance (`dets/`, conf ≥ 0.1), while Ultralytics runs detection again on every frame with its own default threshold.

### Finding 1: the detector is the bottleneck, not the tracker

Both trackers miss about 51,000 people, almost the same number. My v4 is only 0.6 HOTA behind Ultralytics and gives out 446 fewer IDs, which means it switches IDs on the same person less often.

I think the reason is the detector. This YOLOv8m was trained on BDD100K, which is filmed from a camera inside a car. Most of MOT17 is filmed from the street or handheld, so the viewpoint, the size of people and the way they block each other are all different. No matter how I change the tracker, it can't bring back people the detector never found.

### Finding 2: low-score boxes hurt on MOT17-05

On every other sequence v4 is better than v3. MOT17-05 is the only one where it's the other way round: MOTA drops from 38.9 to 21.7, while IDF1 barely changes (49.0 for v4).

MOT17-05 is filmed while walking behind a pedestrian, and people in front often block the people behind them. At first I thought the extra false positives were small people in the distance or cars. But after comparing v4's output boxes with the ground truth one by one, I found that most of the extra boxes are not "empty". They sit on a real person, just in the wrong place (IoU between 0.2 and 0.5):

- 89% of these shifted boxes are shorter than the matching ground-truth box; their median height is only 0.67 of it.
- On average only 46% of the matching person is visible.

The reason is that MOT17 ground-truth boxes cover the whole person, including the hidden part, while the detector can only box the half it can see. A half-body box and a full-body box easily have an IoU below 0.5, and the evaluation punishes it twice: the box itself counts as a false positive, and the real person next to it counts as a miss. v4 picks up low-score boxes to continue tracks, and a lot of what it picks up are exactly these half-body boxes.

This also explains why IDF1 didn't drop with MOTA: IDs actually became more stable (ID switches 241 → 223). The problem is where the boxes are.

## Limitations and next steps

- The detector was trained on car-camera footage, not street footage like MOT17. This is the biggest bottleneck right now.
- Matching is greedy (each track in turn takes the box with the highest IoU), not the globally optimal Hungarian algorithm.
- Velocity prediction is a simple alpha-beta filter, not a Kalman filter.
- The tracker only looks at box positions, not at what people look like, so when two people cross paths their IDs still get swapped easily (964 times across the 7 sequences).
- Next I want to run it on tracking data filmed from a car, such as the MOT part of BDD100K. That would connect it to my previous project and to the driving setting.

## How to run

Data and weights are not in this repo. Download MOT17 from [motchallenge.net](https://motchallenge.net/data/MOT17/) and unzip it to `data/MOT17/`. Download `yolov8m_bdd100k_pedestrian.pt` from the [previous project's release](https://github.com/asiansumer/yolov8-bdd100k-pedestrian/releases), put it in the project root and rename it to `best.pt`. [TrackEval](https://github.com/JonathonLuiten/TrackEval) should sit in a folder next to this project.

Run all commands from the project root:

```bash
# 1. Run YOLOv8m detection on the 7 sequences, output to dets/
python -m scripts.detect_seq

# 2. Run my v3 and v4, output to results/v3 and results/v4
python -m scripts.run_all

# 3. Run Ultralytics' ByteTrack as a reference, output to results/ul_bytetrack
python -m scripts.run_ultralytics_bytetrack

# 4. Evaluate the three trackers with TrackEval
python ../TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train \
  --GT_FOLDER data/MOT17/train --TRACKERS_FOLDER results --SKIP_SPLIT_FOL True \
  --TRACKERS_TO_EVAL v3 v4 ul_bytetrack \
  --SEQ_INFO MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN \
  --METRICS HOTA CLEAR Identity --USE_PARALLEL False --PLOT_CURVES False
```

## Repository layout

```
tracker.py      tracker v1–v4
motion.py       alpha-beta velocity prediction
box_utils.py    IoU and box format conversion
mot_io.py       loading ground truth and detections
scripts/        detection, tracking, reference run, drawing
practice/       small practice scripts
```
