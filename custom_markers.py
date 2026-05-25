import cv2
import numpy as np
import math
DRAW_REF, DRAW_BALL = True, True    

def draw_annotations(img, boxes, cls_ids, names, team_ids, team_colors, scale=2):
    for i, (box, cid) in enumerate(zip(boxes, cls_ids)):
        x1, y1, x2, y2 = map(int, box)
        lbl, xc, yc = names.get(int(cid), '').lower(), (x1 + x2) // 2, (y1 + y2) // 2

        if 'ball' in lbl or 'football' in lbl:
            if DRAW_BALL:
                cv2.fillPoly(img, [np.array([[xc, yc-10], [xc-14, yc-30], [xc+14, yc-30]])], (0, 255, 255))
                cv2.circle(img, (xc, yc), 8, (0, 255, 255), 2, cv2.LINE_AA)
        elif 'referee' in lbl:
            if DRAW_REF:
                ax, ay = min(max(14, (x2-x1)//2+10), 45), min(max(7, (x2-x1)//6+3), 20)
                cv2.ellipse(img, (xc, y2), (ax+5, ay+3), 0, -45, 235, (0, 140, 140), 7, cv2.LINE_AA)
                cv2.ellipse(img, (xc, y2), (ax, ay), 0, -45, 235, (0, 255, 255), 2, cv2.LINE_AA)
        elif (tid := team_ids[i] if i < len(team_ids) else -1) >= 0:
            pts = np.array([[xc, y1-5*scale], [xc-10*scale, y1-20*scale], [xc+10*scale, y1-20*scale]])
            cv2.fillPoly(img, [pts], team_colors[tid])

class TeamTracker:
    def __init__(self, ema_alpha=0.08):
        self.cents, self.alpha = None, ema_alpha

    def assign(self, colors):
        v_idx = [i for i, c in enumerate(colors) if c]
        res, fall = [-1] * len(colors), [(220, 80, 60), (60, 80, 220)]
        if len(v_idx) < 2: return res, fall

        pts = np.array([colors[i] for i in v_idx], dtype=np.float32)
        if self.cents is None:
            _, _, self.cents = cv2.kmeans(pts, 2, None, (3, 20, 0.5), 5, cv2.KMEANS_PP_CENTERS)

        lbls = np.linalg.norm(pts[:, None] - self.cents, axis=2).argmin(axis=1)
        for t in range(2):
            if (m := (lbls == t)).any():
                self.cents[t] = (1 - self.alpha) * self.cents[t] + self.alpha * pts[m].mean(0)
                
        for k, i in enumerate(v_idx): res[i] = int(lbls[k])
        return res, [tuple(map(int, c)) for c in self.cents]

def jersey_color(img, x1, y1, x2, y2):
    if not (crop := img[y1:(y1+y2)//2, x1:x2]).size: return None
    mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), (35, 40, 40), (85, 255, 255)).ravel() == 0
    ng = crop.reshape(-1, 3)[mask]
    return tuple(map(int, ng.mean(0))) if len(ng) >= 20 else None

class PossessionTracker:
    def __init__(self, proximity_px=80):
        self.prox, self.counts = proximity_px, [0, 0]

    def update(self, b_box, p_boxes, t_ids):
        if b_box is None: return -1
        bx, by = (b_box[0] + b_box[2]) / 2, (b_box[1] + b_box[3]) / 2
        
        dist, tid = min(
            ((math.hypot(bx-max(x1,min(x2,bx)), by-max(y1,min(y2,by))), t) 
             for (x1,y1,x2,y2), t in zip(p_boxes, t_ids) if t >= 0), 
            default=(float('inf'), -1)
        )
        if dist <= self.prox:
            self.counts[tid] += 1
            return tid
        return -1

    def percentages(self):
        tot = sum(self.counts)
        return [100 * c / tot for c in self.counts] if tot else [50.0, 50.0]

def draw_possession(img, percs, colors):
    h, w = img.shape[:2]
    bw, bh, by, sx = int(w * 0.4), 20, h - 50, int(w * 0.3)
    w0 = int(bw * percs[0] / 100)
    
    cv2.rectangle(img, (sx, by), (sx + w0, by + bh), colors[0], -1)
    cv2.rectangle(img, (sx + w0, by), (sx + bw, by + bh), colors[1], -1)
    
    cv2.putText(img, f"{int(percs[0])}%", (sx - 50, by + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[0], 2, cv2.LINE_AA)
    cv2.putText(img, f"{int(percs[1])}%", (sx + bw + 10, by + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[1], 2, cv2.LINE_AA)