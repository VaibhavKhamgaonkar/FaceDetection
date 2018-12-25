import cv2
import face_recognition
import dlib
import numpy as np, os,time
import pickle
import pafy
import youtube_dl
import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
from unkownFacesTraning import unknownFacesTraining


# =============================================================================
# #constructing the ArgParser
# =============================================================================

ap = argparse.ArgumentParser()
ap.add_argument('-u','--url', required=True, help ='Enter the youtubbe URL')
ap.add_argument('-t','--thresh', required=False, type= float, default=0.65, help = 'Enter the thresholding value.')
args = vars(ap.parse_args())
# =============================================================================
# Getting BackedupData of Face Encodings
# =============================================================================
def loadEmbeddings(path, file='FaceEncodingsModel.pkl'):
    
    data = [d for d in os.listdir(path) if '.DS_Store' not in d] [0] 
    
    with open(path + file,'rb') as f:
        data = pickle.load(f)
    
    return data 

# =============================================================================
# #running the main file for Face recognition in Video 
# =============================================================================

path = os.path.dirname(os.path.abspath("__file__")) + '/'    
embeddingsPath = path+ 'Encoding_Data/'   
faceModel_path = 'shape_predictor_68_face_landmarks.dat'
faceDbPath = path + 'FacesDB/'

model = loadEmbeddings(embeddingsPath,file='FaceEncodingsModel.pkl')
#getting the people list
person = model.classes_

#======== Not required as Svm Model is trained to do this task========================================
# knownFaces = loadEmbeddings(embeddingsPath) 
# person = list(knownFaces.keys())
# personEncodings = list(knownFaces.values())
# 
# =============================================================================



#gettign the Frontal face area from the image
faceDetector = dlib.get_frontal_face_detector()
#getting landmark Points
landMarker = dlib.shape_predictor(faceModel_path) 
    
    
url = args["url"]#'https://youtu.be/TOu8kEjG1aQ'
#'https://youtu.be/it_aGnqja78' #'https://youtu.be/t27OqUlCSOg' ' #'https://youtu.be/it_aGnqja78'#'https://youtu.be/D03XbsryQF0' #https://youtu.be/zigJlYuxKvo
  #https://youtu.be/t27OqUlCSOg
videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="webm")
global cnt
count=0
#video=cv2.VideoCapture(best.url)
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(best.url) #'/Users/vk250027/Documents/FaceDetection/FDWorking/tests3.mp4
  
cap = FileVideoStream(best.url).start()  
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()
  
print("Input video is laoded... ")
#Get current width of frame
# =============================================================================
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
# #Get current height of frame
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# =============================================================================
#

#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
#writer = cv2.VideoWriter('TestVideo.avi',fourcc, 12.0, (int(450),int(400)))  
writer = None 
print("Entering the loop...")
while cap.more():

    # print("Reading the frame... ")
    frame = cap.read()
    frame = imutils.resize(frame, width=450)
    #s(h, w) = frame.shape[:2]
    #ret,frame = cap.read()
    #print(ret)
    if writer is None:
        (h,w) = frame.shape[:2]
        writer = cv2.VideoWriter('TestVideo.mp4', fourcc, 10.0,
			(w, h), True)
# =============================================================================
#     if (ret == False):
#         break
# =============================================================================
    #print('STarting up')

    try:
        faceLocs = face_recognition.face_locations(frame)
        faceEncodings = face_recognition.face_encodings(frame,faceLocs)
        faces = faceDetector(frame,1)
        print('len of Faces: ', len(faces))
    except Exception as e:
        #print('error')
        print(e)
        continue
    if args['thresh']:
        thresh = args['thresh']
        
    else:
        thresh = 0.65
        
        
    
    cv2.putText(frame, 'Threshold = {}'.format(str(thresh)), (0,50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,0,0),2)
# =============================================================================
#     for face_Encoding in faceEncodings:
# # =============================================================================
# #       # Not needed as we used SVM model for training
#         #result = face_recognition.compare_faces(personEncodings,face_Encoding)
# # =============================================================================
#         #get the index of person having max probability
#         prob = model.predict_proba(np.array(face_Encoding).reshape(1,-1))[0]
#         index = np.argmax(prob)
#         
#         if np.max(prob) > 0.70:
#             text = str(person[index] + str(np.round_(np.max(prob),2)))
#             cv2.rectangle(frame,(x, y), (x+w, y+h), (0,0,255),2)
#             cv2.putText(frame, text, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
#         else :
#             text = 'Unknown'
#             cv2.rectangle(frame,(x, y), (x+w, y+h), (0,255,0),1)
#             cv2.putText(frame, text, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
#             
# =============================================================================
        
    for i in range (0,len(faces)):
        newRect = dlib.rectangle(int(faces[i].left() ),
                                int(faces[i].top() ),
                                int(faces[i].right() ),
                                int(faces[i].bottom() ))
      
        #Getting x,y,w,h coordinate
        x = newRect.left()
        y = newRect.top()
        w = newRect.right() - x
        h = newRect.bottom() - y
        X,Y,W,H = x,y,w,h
        
        
        prob = model.predict_proba(np.array(faceEncodings[i]).reshape(1,-1))[0]
        index = np.argmax(prob)
        
        if np.max(prob) > float(thresh):
            text = str(person[index] + str(np.round_(np.max(prob),2)))
            cv2.rectangle(frame,(x, y), (x+w, y+h), (0,255,0),2)
            cv2.putText(frame, text, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        else :
            text = 'Unknown'
            cv2.rectangle(frame,(x, y), (x+w, y+h), (0,0,255),1)
            cv2.putText(frame, text, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            
# =============================================================================
#             Capturing the Unkown data for next lever of self training 
# =============================================================================
            if not os.path.exists(faceDbPath + 'Unknown'):
                os.mkdir(faceDbPath + 'Unknown')
            
            cv2.imwrite(faceDbPath+'Unknown/'+ str(count) + '.jpg', frame[y:y+h,x:x+w])
            count+=1
            
            
              #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
              
              #creating backup of coordinates
              
        
# =============================================================================
#         #checking unknown user first
#         
#         if  True not in result:
#             text = 'Unknown Person'
#             print(text)
#         else:
#             #getting the name of person
#             text = person[int(np.argmax(result))]
#             print(text)
#             
# =============================================================================
        #Now Drawing the box and tagging the person 
        
# =============================================================================
#         for (top,right,bottom,left) in faceLocs:
#             
#             if text == 'Unknown':
#                 cv2.rectangle(frame,(left, top), (right, bottom), (0,0,255),2)
#                 cv2.putText(frame, text, (left,top),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
#             elif text == 'Not Required':
#                 pass
#             
#             else:
#                 cv2.rectangle(frame,(left, top), (right, bottom), (0,255,0),1)
#                 cv2.putText(frame, text, (left,top),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
#     
# =============================================================================
    
    cv2.imshow('frame',frame)
    writer.write(frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break    
    #if cv2.waitKey(20) & 0xFF == ord('q'):
    #   break
    
#cap.release()
cv2.destroyAllWindows()
cap.stop()
writer.release()
choice = 'y'
while choice != 'n':
    choice = str(input('Would you like to check the Unknown faces ? y/n ..\n', ))
    if choice.lower() =='y':
        uft = unknownFacesTraining()
        uft.playWithUnknownImages()
        break
    

    
