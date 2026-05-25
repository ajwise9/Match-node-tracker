class IdTracker:
    def __init__(self, iou_thresh=0.3, max_age=20):
        self.t, self.mx, self.tr, self.nid = iou_thresh, max_age, {}, 0

    def _iou(self, a, b):
        x1, y1, x2, y2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
        i = max(0, x2-x1) * max(0, y2-y1)
        u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
        return i/u if u > 0 else 0

    def update(self, boxes, mask):
        ids, used = [-1] * len(boxes), set()
        for i, box in enumerate(boxes):
            if not mask[i]: continue
            best, bv = -1, self.t
            for tid, (tbox, _) in self.tr.items():
                if tid not in used and (v := self._iou(box, tbox)) > bv: best, bv = tid, v
            if best < 0: best, self.nid = self.nid, self.nid + 1
            ids[i] = best; used.add(best); self.tr[best] = [box, 0]
        for tid in list(self.tr):
            if tid not in used:
                self.tr[tid][1] += 1
                if self.tr[tid][1] > self.mx: del self.tr[tid]
        return ids