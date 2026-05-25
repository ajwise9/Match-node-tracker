import math

class PossessionTracker:
    def __init__(self, proximity_px=80):
        self.prox, self.counts = proximity_px, [0, 0]

    def update(self, b_box, p_boxes, t_ids):
        if not b_box: return -1
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