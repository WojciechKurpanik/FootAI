from supervision import ByteTrack
from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = model_path
        self.tracker = sv.ByteTrack()

    def in_frames_detection(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection_batch

        return detections

    def track_objects_in_frames(self, frames):

        detections = self.in_frames_detection(frames)

        tracked_objects = {
            "players":[],
            "goalkeepers":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names

            detection_sv = sv.Detections.from_ultralytics(detection)

            add_tracker = self.tracker.update_with_detections(detection_sv)

            # dodawanie bbox'ów danej klasy do listy
            tracked_objects["players"].append({})
            tracked_objects["goalkeepers"].append({})
            tracked_objects["referees"].append({})
            tracked_objects["ball"].append({})

            for frame_detection in add_tracker:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names['player']:
                    tracked_objects["players"][frame_num][track_id] = {"bbox":bbox}

                if class_id == class_names['referee']:
                    tracked_objects["referees"][frame_num][track_id] = {"bbox":bbox}

                if class_id == class_names['goalkeeper']:
                    tracked_objects["goalkeepers"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_names['ball']:
                    tracked_objects["ball"][frame_num][1] = {"bbox":bbox}

        return tracked_objects