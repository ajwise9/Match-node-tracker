import cv2, numpy as np
from collections import deque


class SpeedTracker:
    def __init__(self, fps=30.0, h_m=1.80, window=8, alpha=0.4, max_kmh=40.0, deadband=2.0):
        self.fps, self.h_m, self.win, self.a, self.mx, self.db = fps, h_m, window, alpha, max_kmh, deadband
        self.prev, self.cum, self.hist, self.sm, self.f = None, np.eye(3, dtype=np.float32), {}, {}, 0

    def _cam(self, frame, boxes):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev is not None:
            H, W = g.shape; m = np.full((H, W), 255, np.uint8)
            for x1, y1, x2, y2 in boxes:
                m[max(0,int(y1)-15):min(H,int(y2)+15), max(0,int(x1)-15):min(W,int(x2)+15)] = 0
            p = cv2.goodFeaturesToTrack(self.prev, 300, 0.01, 10, mask=m, blockSize=7)
            if p is not None and len(p) >= 10:
                q, s, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p, None); ok = s.ravel().astype(bool)
                if ok.sum() >= 10 and (M := cv2.estimateAffinePartial2D(q[ok], p[ok], method=cv2.RANSAC)[0]) is not None:
                    self.cum = self.cum @ np.vstack([M, [0, 0, 1]]).astype(np.float32)
        self.prev = g

    def update(self, frame, boxes, ids):
        self._cam(frame, boxes); out = [None] * len(boxes)
        for i, (box, tid) in enumerate(zip(boxes, [int(t) for t in ids])):
            if tid < 0: continue
            x1, y1, x2, y2 = box; bh = max(float(y2-y1), 1.0); mpp = self.h_m / bh
            wx, wy, _ = self.cum @ np.array([(x1+x2)/2, y2, 1.0], np.float32)
            h = self.hist.setdefault(tid, deque(maxlen=self.win))
            h.append((float(wx), float(wy), self.f, mpp))
            if len(h) < 3 or (dt := (h[-1][2] - h[0][2]) / self.fps) <= 0: continue
            amp = sum(p[3] for p in h) / len(h)
            kmh = min(self.mx, np.hypot(h[-1][0]-h[0][0], h[-1][1]-h[0][1]) * amp / dt * 3.6)
            kmh = self.a * kmh + (1 - self.a) * self.sm.get(tid, kmh)
            self.sm[tid] = kmh = 0.0 if kmh < self.db else kmh
            out[i] = kmh
        self.f += 1
        return out


def draw_speed(img, boxes, speeds, team_ids=None, team_colors=None, min_show=1.0):
    for i, (box, sp) in enumerate(zip(boxes, speeds)):
        if sp is None or sp < min_show: continue
        x1, y1, x2, y2 = map(int, box); txt = f"{sp:.1f} km/h"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        bw, bh = tw + 12, th + 8
        bx, by = (x1 + x2) // 2 - bw // 2, y2 + 10
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
        cv2.putText(img, txt, (bx + 6, by + bh - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)