import cv2
import numpy as np
from .pitch_configuration import SoccerPitchConfiguration


class ViewTransformer:
    def __init__(self, config: SoccerPitchConfiguration, scale=0.1):
        self.scale = scale
        self.config = config

        # Generowanie mapowania automatycznie z klasy config
        # mapujemy indeksy vertex -> pixel coords (x_px, y_px)
        self.keypoint_mapping = {}
        for i, vertex in enumerate(config.vertices):
            # Zamiana cm -> px
            x_px = int(vertex[0] * self.scale)
            y_px = int(vertex[1] * self.scale)
            self.keypoint_mapping[i] = (x_px, y_px)

        print(f"[ViewTransformer] Załadowano {len(self.keypoint_mapping)} punktów z konfiguracji boiska.")

    def _extract_keypoints_from_result(self, keypoint_detections):
        """
        Obsługuje różne typy wyników z ultralytics:
        - jeśli model zwraca .keypoints -> użyj ich (pose)
        - jeśli model zwraca .boxes z .cls i .xywh -> interpretuj to jako keypoint boxes
        Zwraca listę tupli: (class_id, x_center, y_center)
        """
        kp_list = []
        if keypoint_detections is None:
            return kp_list

        # 1) ultralytics: results.keypoints (pose task) -> list of keypoints per instance
        if hasattr(keypoint_detections, "keypoints") and keypoint_detections.keypoints is not None:
            try:
                # keypoints może być Tensor NxKx3 lub lista; spróbuj bezpiecznie odczytać
                kps = keypoint_detections.keypoints
                # jeśli kps ma strukturę boxes-like (np. .xy), obsłuż to inaczej
                if hasattr(kps, "xy"):
                    # może być kps.xy - lista punktów
                    arr = kps.xy
                    for i, single in enumerate(arr):
                        # single może zawierać wiele punktów; tutaj nie mamy klasy -> skip
                        pass
                else:
                    # rzutujemy na numpy jeśli to tensor
                    try:
                        kparr = np.asarray(kps)
                        # format zależny od modelu - nie przewidujemy na stałe, więc jedynie debug
                        # nie próbujemy mapowania klas tutaj, tylko zwracamy puste - dalsza logika używa boxes jeśli da się
                    except Exception:
                        pass
            except Exception:
                pass

        # 2) standardowe boxes (użyte w Twojej poprzedniej implementacji)
        if hasattr(keypoint_detections, "boxes") and keypoint_detections.boxes is not None:
            boxes = keypoint_detections.boxes
            # klasy
            try:
                cls_tensor_list = boxes.cls
                xywh_list = boxes.xywh
                for i in range(len(cls_tensor_list)):
                    try:
                        class_id = int(cls_tensor_list[i].item())
                        xywh = xywh_list[i].cpu().numpy()
                        x_c, y_c = float(xywh[0]), float(xywh[1])
                        kp_list.append((class_id, x_c, y_c))
                    except Exception:
                        continue
            except Exception:
                # jeśli boxes posiada inne pola, próbuj xyxy centroid
                try:
                    xyxy_list = boxes.xyxy
                    for i in range(len(xyxy_list)):
                        xyxy = xyxy_list[i].cpu().numpy()
                        x_c = float((xyxy[0] + xyxy[2]) / 2.0)
                        y_c = float((xyxy[1] + xyxy[3]) / 2.0)
                        class_id = int(boxes.cls[i].item()) if hasattr(boxes, "cls") else i
                        kp_list.append((class_id, x_c, y_c))
                except Exception:
                    pass

        # 3) jeśli nic nie znaleziono - zwróć pustą listę
        return kp_list

    def transform_points(self, player_detections, keypoint_detections):
        """
        player_detections: lista [(x, y, team_id), ...] - x,y w pikselach obrazu wejściowego
        keypoint_detections: wynik modelu YOLO z kluczowymi punktami (ultralytics result)
        Zwraca listę: [(x_on_pitch_px, y_on_pitch_px, team_id), ...]
        """
        # Wyciągamy keypointy z detekcji w ustrukturyzowanej formie (class_id, x, y)
        kp_centers = self._extract_keypoints_from_result(keypoint_detections)

        src_pts = []
        dst_pts = []

        # Zbuduj src/dst na podstawie mapowania class_id -> vertex
        for class_id, x_c, y_c in kp_centers:
            if class_id in self.keypoint_mapping:
                src_pts.append([x_c, y_c])
                dst_pts.append(list(self.keypoint_mapping[class_id]))

        print(f"[ViewTransformer] Detected keypoint matches: src_pts={len(src_pts)}, dst_pts={len(dst_pts)}")

        transformed_players = []

        if len(src_pts) >= 4:
            # homografia
            src_pts_np = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
            dst_pts_np = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)

            h_matrix, mask = cv2.findHomography(src_pts_np, dst_pts_np, cv2.RANSAC, 5.0)
            if h_matrix is None:
                print("[ViewTransformer] findHomography zwrócił None")
            else:
                print(f"[ViewTransformer] Homography OK, matrix shape: {h_matrix.shape}")

                if player_detections and len(player_detections) > 0:
                    points_np = np.array([[p[0], p[1]] for p in player_detections], dtype=np.float32).reshape(-1, 1, 2)
                    transformed_np = cv2.perspectiveTransform(points_np, h_matrix).reshape(-1, 2)

                    max_x = self.config.length * self.scale
                    max_y = self.config.width * self.scale

                    for i, (tx, ty) in enumerate(transformed_np):
                        team_id = player_detections[i][2]
                        # Inwersja osi Y (bo matplotlib/plt.imshow używa top-left origin by default)
                        ty_inv = max_y - ty
                        if 0 <= tx <= max_x and 0 <= ty_inv <= max_y:
                            transformed_players.append((float(tx), float(ty_inv), int(team_id)))

        elif len(src_pts) >= 3:
            # fallback: estimateAffinePartial2D (rotacja + skalowanie + translacja)
            src_np = np.array(src_pts, dtype=np.float32)
            dst_np = np.array(dst_pts, dtype=np.float32)
            # estimateAffinePartial2D expects shape (N,2)
            M, inliers = cv2.estimateAffinePartial2D(src_np, dst_np, method=cv2.RANSAC)
            if M is None:
                print("[ViewTransformer] estimateAffinePartial2D zwrócił None")
            else:
                print("[ViewTransformer] Affine fallback used")
                if player_detections and len(player_detections) > 0:
                    pts = np.array([[p[0], p[1]] for p in player_detections], dtype=np.float32)
                    # dodaj jedynki do mnożenia
                    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
                    pts_h = np.hstack([pts, ones])
                    transformed = (M @ pts_h.T).T  # shape (N,2)

                    max_x = self.config.length * self.scale
                    max_y = self.config.width * self.scale

                    for i, (tx, ty) in enumerate(transformed):
                        team_id = player_detections[i][2]
                        ty_inv = max_y - ty
                        if 0 <= tx <= max_x and 0 <= ty_inv <= max_y:
                            transformed_players.append((float(tx), float(ty_inv), int(team_id)))

        else:
            # za mało keypointów do transformacji
            print("[ViewTransformer] Za mało punktów keypoint do estymacji transformacji (wymagane >=3).")

        print(f"[ViewTransformer] Transformed players count: {len(transformed_players)}")
        return transformed_players
