import cv2
from FaceDetection.face_detection import face
from keras.models import load_model
import numpy as np
import os
from embedding import emb
from MongoDB.retrieve_pymongo_data import database

import warnings
warnings.filterwarnings("ignore")

from tensorflow import ConfigProto, Session
from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import gc

# Reset Keras Session 
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try :
         del classifier # this is from global space - change this as you need 
    except :
         pass

    print (gc.collect()) # if it does something you should see a number as output

    # use the same config as you used to create the session 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1 
    config.gpu_options.visible_device_list = " 0 " 
    set_session(Session(config =config))


def Recognition(subject):
    label=None

    people=sorted(os.listdir('people'))


    lecture=subject


    person=None
    
    def Create_labels():
        people=sorted(os.listdir('people'))
        students={}
        attendance_count={}
        for i in people:
            students[(int(i[0])-1)]=i[1:]
            attendance_count[(int(i[0])-1)]=0
        return students,attendance_count


    people,attendance_count=Create_labels()
    completed_label="Attendance is Completed"
    reset_keras()
    e=emb()
    fd=face()

    model=load_model('Model/Face_recognition.MODEL')


    data=database() ##### Intitalising the Mongo Database

    color=(0, 255, 0) 
    cap=cv2.VideoCapture(0)
    ret=True

    while ret:
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
        det,coor=fd.detectFace(frame)

        if(det is not None):
            for i in range(len(det)):
                detected=det[i]
                k=coor[i]
                f=detected
                detected=cv2.resize(detected,(160,160))
                detected=detected.astype('float')/255.0
                detected=np.expand_dims(detected,axis=0)
                feed=e.calculate(detected)
                feed=np.expand_dims(feed,axis=0)
                prediction=model.predict(feed)[0]

                result=int(np.argmax(prediction))

                if(np.max(prediction)>.85):
                    for i in people:
                        if(result==i):
                            label=people[i]
                            
                            if(attendance_count[i]<30):
                                attendance_count[i]=attendance_count[i]+1
                                # MONGO LINE HERE
                                if(attendance_count[i]==30): data.update(label,lecture)
                            person=i
                else:
                    label='unknown'

                try:
                    if(int(attendance_count[person])>=30):   
                        cv2.putText(frame,completed_label,(k[0],k[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        cv2.rectangle(frame,(k[0],k[1]),(k[0]+k[2],k[1]+k[3]),(0,255,0),3)
                    else:
                        cv2.putText(frame,label,(k[0],k[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        cv2.rectangle(frame,(k[0],k[1]),(k[0]+k[2],k[1]+k[3]),(255,0,0),3)
                except:
                    pass

                
        cv2.imshow('Say Cheese and Press "Q" to Quit',frame)
        if(cv2.waitKey(1) & 0XFF==ord('q')): break
        if cv2.getWindowProperty('Say Cheese and Press "Q" to Quit', cv2.WND_PROP_VISIBLE) < 1: break

    cap.release()
    cv2.destroyAllWindows()
    data.export_csv(lecture) ###
    return
