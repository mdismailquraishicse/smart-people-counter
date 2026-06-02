import cv2
from ultralytics import YOLO



class PeopleCounter:


    def __init__(self):


        self.SOURCE = 0
        self.model = YOLO("yolov8n.pt")
        self.count: dict = {
            "up": 0,
            "down": 0,
            "left": 0,
            "right": 0
        }
        
        self.LINE_X:int = None
        self.LINE_Y:int = None
        self.FRAME_SIZE = (1280, 620)


    def fun(self):

        cap = cv2.VideoCapture(self.SOURCE)
        while True:
            ret, frame = cap.read()
            if self.SOURCE == 0:
                frame = cv2.flip(frame, 1)

            if not ret:
                break

            frame = cv2.resize(frame, (self.FRAME_SIZE))
            h,w = frame.shape[:2]
            self.LINE_X = w // 2
            self.LINE_Y = h // 2

            results = self.model.track(
                frame,
                persist = True,
                verbose = False
            )
            annotated = results[0].plot()




            cv2.line(
                frame,
                (0, self.LINE_Y),
                (w, self.LINE_Y),
                (0, 255, 0),
                2
            )
            cv2.imshow("PeopleCounter", annotated)
            if cv2.waitKey(1) == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    pc = PeopleCounter()
    pc.fun()