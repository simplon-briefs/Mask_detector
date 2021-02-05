import cv2
from keras.models import load_model
import numpy as np
import playsound
import time
#import vlc
from sklearn.metrics import accuracy_score
from keras_mask_detector import Keras_mask_detector


class Video_mask_detector:
    def __init__(self):
        self.model = None
        self.haar = cv2.CascadeClassifier("../xml/haarcascade_frontalface_default.xml")
        self.get_model()
        self.video()

    def get_model(self):
        try:
            self.model = load_model("../model.h5")
        except:
            Keras_mask_detector()
            self.get_model()

    def detect_face_mask(self, img):
        y_pred = self.model.predict_classes(img.reshape(1,224,224,3))
        yy= self.model.predict(img.reshape(1,224,224,3))
        return y_pred[0][0],yy
    
    def draw_label(self, img,text,pos,bg_color):
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

    def song(self):
        p = vlc.MediaPlayer("song/song.mp3")
        p.play()
        time.sleep(1)
        p.stop()

    def detect_face(self, img):   
        coods = self.haar.detectMultiScale(img)
        return coods

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