import os; os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse, cv2, torch
from ultralytics import YOLO
from custom_markers import draw_annotations, jersey_color, TeamTracker, PossessionTracker, draw_possession
import numpy as np
from speed_tracker import SpeedTracker, draw_speed
from id_tracker import IdTracker

SPEED_BALL, SPEED_REF = False, False  


p = argparse.ArgumentParser()
p.add_argument('--model', default='train/weights/best.pt')
p.add_argument('--source', default='media/test.mkv')
p.add_argument('--stream', action='store_true')
p.add_argument('--device', default=None)
p.add_argument('--imgsz', type=int, default=640)
p.add_argument('--conf', type=float, default=0.25)
a = p.parse_args()

a.device = a.device or ('mps' if torch.backends.mps.is_available() else '0' if torch.cuda.is_available() else 'cpu')
src = a.source.lstrip('usb') if a.source.startswith('usb') else a.source
is_img = src.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))
if not is_img and not a.stream: a.stream = input('Stream preview? [y/N]: ').strip().lower() == 'y'

os.makedirs('output', exist_ok=True)
fps = 30.0
if not is_img: c = cv2.VideoCapture(src); fps = c.get(cv2.CAP_PROP_FPS) or 30.0; c.release()
writer, frames = None, 0

tracker, pos_tracker = TeamTracker(), PossessionTracker()
spd_tracker, id_tracker = SpeedTracker(fps=fps), IdTracker()

for r in YOLO(a.model).predict(source=src, stream=True, device=a.device, half=True, imgsz=a.imgsz, conf=a.conf, verbose=False):
    img = r.orig_img.copy()
    b, c, n = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.names

    colors = [jersey_color(img, int(x[0]), int(x[1]), int(x[2]), int(x[3])) for x in b]
    team_ids, team_colors = tracker.assign(colors)

    ball = next((x for x, cid in zip(b, c) if 'ball' in n[int(cid)].lower()), None)
    pos_tracker.update(ball, b, team_ids)

    is_player = [(SPEED_BALL if ('ball' in (lbl := n[int(cid)].lower()) or 'football' in lbl)
                  else SPEED_REF if 'referee' in lbl
                  else team_ids[i] >= 0)
                 for i, cid in enumerate(c)]
    ids = id_tracker.update(b, is_player)
    speeds = spd_tracker.update(img, b, ids)

    draw_annotations(img, b, c, n, team_ids, team_colors)
    draw_possession(img, pos_tracker.percentages(), team_colors)
    draw_speed(img, b, speeds, team_ids, team_colors)
   
    if is_img:
        cv2.imwrite('output/result.jpg', img); print('✓ Saved → output/result.jpg'); break

    if not writer: writer = cv2.VideoWriter('output/result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (img.shape[1], img.shape[0]))
    writer.write(img); frames += 1

    if a.stream:
        cv2.imshow('Detection', img)
        if cv2.waitKey(1) == ord('q'): break
    elif frames % 30 == 0: print(f'frames: {frames}', end='\r')

if writer: writer.release(); print(f'\n✓ Video saved → output/result.mp4 ({frames} frames)')
cv2.destroyAllWindows()