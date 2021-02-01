import cv2
from keras.models import load_model

from keras_mask_detector import Keras_mask_detector


class Video_mask_detector:
    def __init__(self):
        self.model = None

        self.get_model()
        self.video()

    def get_model(self):
        try:
            self.model = load_model("model/model.h5")
        except:
            Keras_mask_detector()
            self.get_model()

    def detect_face_mask(self, img):
        y_pred = self.model.predict_classes(img.reshape(1,224,224,3))
        return y_pred[0][0]
    
    def draw_label(self, img,text,pos,bg_color):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
 
        end_X = pos[0] + text_size[0][0] + 2
        end_y = pos[1] + text_size[0][1] - 2
        
        cv2.rectangle(img,pos,(end_X,end_y), bg_color,cv2.FILLED)
        cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

    def video(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret,frame = cap.read()
            frame = cv2.resize(frame,(224,224))
            
            y_pred = detect_face_mask(frame)
            
            print(y_pred)
            if y_pred == 0:
                self.draw_label(frame,"Mask", (30,30),(0,255,0))
            else:
                self.draw_label(frame, "No Mask", (30,30),(0,0,255))
                    
            cv2.imshow("window", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
        cv2.destroyAllWindows()

Video_mask_detector()