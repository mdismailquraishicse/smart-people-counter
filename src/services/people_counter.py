import cv2
from ultralytics import YOLO
from core.config import settings


class PeopleCounter:


    def __init__(self):

        self.SOURCE = settings.SOURCE
        self.model = YOLO(settings.MODEL)

        self.count: dict = {
            "up": 0,
            "down": 0,
            "left": 0,
            "right": 0
        }

        self.FRAME_SIZE = (1280, 620)
        self.LINE_X:int = None
        self.LINE_Y:int = None
        self.track_history_x = {}
        self.track_history_y = {}
        self.crossed_ud_ids = set()
        self.crossed_lr_ids = set()


    @property
    def total_in_frame(self):

        return self.count.get("up") + self.count.get("down")


    def get_coordinates(self, boxes):

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xywhs = boxes.xywh.cpu().numpy()
            return zip(ids, xywhs)
        return []
        

    def update_crosser(self, tid:int, crossed_ids, line, track_history:dict, c, direction_1:str, direction_2:str):
        
        if tid in track_history and tid not in crossed_ids:
            prev = track_history[tid]
            if prev < line <=c:
                self.count[direction_1] += 1
                crossed_ids.add(tid)
            elif prev > line >= c:
                self.count[direction_2] += 1
                crossed_ids.add(tid)
        track_history[tid] = c


    def crosser_counter(self, cap):

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if self.SOURCE == 0:
                frame = cv2.flip(frame, 1)

            frame = cv2.resize(frame, (self.FRAME_SIZE))
            h,w = frame.shape[:2]
            self.LINE_X = w // 2
            self.LINE_Y = h // 2

            results = self.model.track(
                frame,
                persist = True,
                verbose = False,
                classes = [0]
            )
            boxes = results[0].boxes
            co_ordinates = self.get_coordinates(boxes = boxes)
            for tid, xywh in co_ordinates:
                cx = int(xywh[0])
                cy = int(xywh[1])
                self.update_crosser(tid=tid, crossed_ids=self.crossed_ud_ids, line=self.LINE_Y,
                                    track_history=self.track_history_y, c=cy, direction_1="down", direction_2="up")
                self.update_crosser(tid=tid, crossed_ids=self.crossed_lr_ids, line=self.LINE_X,
                                    track_history= self.track_history_x,c=cx, direction_1="right", direction_2="left")

                cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
            cv2.line(frame, (0, self.LINE_Y), (w, self.LINE_Y), (0, 255, 0), 2)
            cv2.line(frame, (self.LINE_X, 0), (self.LINE_X, h), (0, 255, 0), 2)
            cv2.putText(frame, f"up: {self.count.get('up')}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0), 1)
            cv2.putText(frame, f"down: {self.count.get('down')}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0), 1)
            cv2.putText(frame, f"left: {self.count.get('left')}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0), 1)
            cv2.putText(frame, f"right: {self.count.get('right')}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0), 1)
            cv2.imshow("PeopleCounter", frame)
            if cv2.waitKey(1) == ord("q"):
                break
