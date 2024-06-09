from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np



class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'model_data/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
        self.id_mapping = {}
    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()


    def get_tracks(self):

        for tracks in self.tracker.tracks:
            print("Tracks: " + str(tracks.track_id))

    def update_tracks(self, new_id= None, track_to_change = None ): 
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if new_id != None:
                if track.track_id not in self.id_mapping:
                    self.id_mapping[track.track_id] = new_id
                if track_to_change != None:
                    self.id_mapping[track_to_change] = new_id

            
            bbox = list(track.to_tlbr())

            id = track.track_id

            tracks.append(Track(id, bbox))

        self.tracks = tracks
    
    def get_id(self, id):

        if id in self.id_mapping:
            return self.id_mapping[id]
        else:
            return None


class Track:
    track_id = None
    bbox = None
    reidentified = None
    def __init__(self, id, bbox, ):
        self.track_id = id
        self.bbox = bbox

        