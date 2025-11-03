# from .kalman_filter import KalmanBoxTracker
# import numpy as np

# class Sort:
#     def __init__(self, max_age=30, min_hits=2, iou_threshold=0.3):
#         self.max_age = max_age
#         self.min_hits = min_hits
#         self.iou_threshold = iou_threshold
#         self.trackers = []
#         self.frame_count = 0

#     def update(self, detections):
#         self.frame_count += 1

#         trks = np.zeros((len(self.trackers), 5))
#         ret = []

#         for t, trk in enumerate(self.trackers):
#             pos = trk.predict()[0]
#             trks[t][:4] = pos.ravel()
#             trks[t][4] = 0

#         matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)

#         # Update matched trackers with corresponding detections
#         for det_idx, trk_idx in matched:
#             self.trackers[trk_idx].update(detections[det_idx])

#         # Create and initialise new trackers for unmatched detections
#         for i in unmatched_dets:
#             trk = KalmanBoxTracker(detections[i])
#             self.trackers.append(trk)

#         # Prepare output for confirmed trackers
#         for trk in self.trackers:
#             d = trk.get_state()[0]
#             if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
#                 ret.append(np.concatenate((d.ravel(), [trk.id])).reshape(1, -1))

#         # Remove dead tracklets
#         self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]

#         if len(ret) > 0:
#             return np.concatenate(ret)
#         return np.empty((0, 5))

#     def associate_detections_to_trackers(self, detections, trackers):
#         if len(trackers) == 0:
#             return np.empty((0, 2), dtype=int), np.arange(len(detections)), []

#         iou_matrix = self.iou_batch(detections, trackers)

#         matched_indices = []
#         unmatched_detections = []
#         unmatched_trackers = []

#         for d in range(len(detections)):
#             best_match = np.argmax(iou_matrix[d])
#             if iou_matrix[d][best_match] < self.iou_threshold:
#                 unmatched_detections.append(d)
#             else:
#                 matched_indices.append([d, best_match])

#         for t in range(len(trackers)):
#             if t not in [m[1] for m in matched_indices]:
#                 unmatched_trackers.append(t)

#         return np.array(matched_indices), np.array(unmatched_detections), unmatched_trackers

#     def iou_batch(self, boxes1, boxes2):
#         iou_matrix = np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

#         for i, box1 in enumerate(boxes1):
#             for j, box2 in enumerate(boxes2):
#                 iou_matrix[i, j] = self.iou(box1, box2)

#         return iou_matrix

#     def iou(self, bb_test, bb_gt):
#         xx1 = np.maximum(bb_test[0], bb_gt[0])
#         yy1 = np.maximum(bb_test[1], bb_gt[1])
#         xx2 = np.minimum(bb_test[2], bb_gt[2])
#         yy2 = np.minimum(bb_test[3], bb_gt[3])
#         w = np.maximum(0., xx2 - xx1)
#         h = np.maximum(0., yy2 - yy1)
#         wh = w * h
#         o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
#                   + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
#         return o


import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian algorithm

from .kalman_filter import KalmanBoxTracker


class Sort:
    def __init__(self, max_age=30, min_hits=2, iou_threshold=0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        ret = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t][:4] = pos.ravel()
            trks[t][4] = 0  # no score for prediction

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(detections, trks)

        # Update matched trackers with corresponding detections
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)

        # Prepare output for confirmed trackers
        for trk in self.trackers:
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d.ravel(), [trk.id])).reshape(1, -1))

        # Remove dead tracklets
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), []

        iou_matrix = self.iou_batch(detections, trackers)

        # Hungarian algorithm finds minimum cost assignment,
        # so convert IOU to cost by negative sign
        cost_matrix = -iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_indices = []
        unmatched_detections = []
        unmatched_trackers = []

        for d, t in zip(row_ind, col_ind):
            if iou_matrix[d, t] < self.iou_threshold:
                unmatched_detections.append(d)
                unmatched_trackers.append(t)
            else:
                matched_indices.append([d, t])

        unmatched_detections += [d for d in range(len(detections)) if d not in row_ind]
        unmatched_trackers += [t for t in range(len(trackers)) if t not in col_ind]

        return np.array(matched_indices), np.array(unmatched_detections), unmatched_trackers

    def iou_batch(self, boxes1, boxes2):
        iou_matrix = np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self.iou(box1, box2)

        return iou_matrix

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        o = wh / (
            (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh
        )
        return o
