import cv2
import numpy as np
import dlib

detector=dlib.get_frontal_face_detector()
#setup dlib face predictor 
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


cap=cv2.VideoCapture(0);
while(True):
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=detector(gray)
    d=0
    for i in face:
        landmark=predictor(gray,i)
        landmarkpoint=[]
        for j in range(0,68):
            x_point=landmark.part(j).x
            y_point=landmark.part(j).y
            landmarkpoint.append((x_point,y_point))
        maxx=0
        minx=1000000
        miny=1000000
        maxy=0
        for k in range(0,len(landmarkpoint)):
            if(landmarkpoint[k][0]<minx):
                minx=landmarkpoint[k][0]
            if(landmarkpoint[k][0]>maxx):
                maxx=landmarkpoint[k][0]
            if(landmarkpoint[k][1]<miny):
                miny=landmarkpoint[k][1]
            if(landmarkpoint[k][1]>maxy):
                maxy=landmarkpoint[k][1]
        box=np.array([(minx,maxy),(minx,miny),(maxx,miny),(maxx,maxy)])
        cv2.polylines(frame,[box],True,(255,0,0),3)
    cv2.imshow("result",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

