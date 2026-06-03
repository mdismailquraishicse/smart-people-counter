import cv2
from services.people_counter import PeopleCounter

if __name__ == "__main__":

    pc = PeopleCounter()
    cap = cv2.VideoCapture(pc.SOURCE)
    pc.crosser_counter(cap = cap)
    cap.release()
    cv2.destroyAllWindows()