import cv2
from keras.models import load_model

from keras_mask_detector import Keras_mask_detector


class Video_mask_detector:
    def __init__(self):
        self.model = None

        self.get_model()
        self.Video()

    def get_model(self):
        try:
            self.model = load_model("model/model.h5")
        except:
            Keras_mask_detector()
            self.get_model()

    def detect_face_mask(self, img):
        y_pred = self.model.predict_classes(img.reshape(1,224,224,3))
        return y_pred[0][0]

    def Video(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret,frame = cap.read()
            frame = cv2.resize(frame,(224,224))
            
            y_pred = detect_face_mask(frame)
            
            print(y_pred)
            
            cv2.imshow("window", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
        cv2.destroyAllWindows()

Video_mask_detector()