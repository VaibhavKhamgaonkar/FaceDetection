# =============================================================================
# Face Detection using Face Alignment wrap affine method
# =============================================================================

import numpy as np, os,time 
import cv2
import dlib
#import openface
import face_recognition
import pickle
from sklearn.svm import SVC


class FaceTraining():
    def __init__(self):
        self.path = os.path.dirname(os.path.abspath("__file__")) + '/'
        self.imageDir = self.path + 'FacesDB/'
        self.knownFaces = []
        self.knownEncodings =[]
        self.count=0
        self.folders = os.listdir(self.imageDir)
        self.folders = [item for item in self.folders if item != '.DS_Store']
        self.folders = [item for item in self.folders if item != 'Unknown']
        self.svm = SVC(C=100.0,kernel='linear',gamma=0.001,probability=True,verbose=2)


    def storeData(self):
    
        # =============================================================================
        # Storing the Encodings Data
        # =============================================================================
        with open(self.path+ 'Encoding_Data/' + 'FaceEncodings.pkl','wb') as f:
                        pickle.dump(self.knownFaces, f,  protocol=2)
        # =============================================================================
        # Storing the model.
        # =============================================================================
        with open(self.path+ 'Encoding_Data/' + 'FaceEncodingsModel.pkl','wb') as f:
                        pickle.dump(self.svm, f,  protocol=2)
        
        
        print("Training Finished...! Model can be found at {} location ".format(self.path+ 'Encoding_Data/'))
    

    # =============================================================================
    # Function to train on Data
    # =============================================================================
    def train(self):
        for folder in self.folders:
            #knownFaces.append(folder)
            images = [item for item in os.listdir(self.imageDir + folder +'/') if item != '.DS_Store']
            
            print('Starting {} which has {} Images'.format(folder,len(images)))
            
            for img in images:   
                image = cv2.imread(self.imageDir + folder +'/' + img)
                try:    
                #getting the face location
                    faceLcoation = face_recognition.face_locations(image)
                    #now getting faceEncodings
                    faceEncodings = face_recognition.face_encodings(image,faceLcoation)[0]    
                    #knownFaces[str(folder)] = faceEncodings
                except:
                    #print('count:' , count)
                    self.count+=1
                    continue
                self.knownFaces.append(folder)
                self.knownEncodings.append(faceEncodings)
            
            print('done with {} and number of not found Faces: {} out of total {} facesData '.format(folder, self.count, len(images)))
            self.count = 0
        
        
        # =============================================================================
        # training the SVM 
        # =============================================================================
        #Training the SVM
        print("Training Started...")
        
        self.svm.fit(self.knownEncodings,self.knownFaces)
        
        #calling the function to store the Model and other data
        FaceTraining.storeData(self)
        print('Training finished')



if __name__ == '__main__':
    obj = FaceTraining()
    obj.train()


# =============================================================================
# Lets test it
# =============================================================================
# =============================================================================
# temp = list(knownFaces.values())
# person = list(knownFaces.keys())
#     
# img = cv2.imread('/Users/vk250027/Documents/FaceDetection/FaceDetectionFinal/FacesDB/Bastian S/19.jpg')    
# fLoc = face_recognition.face_locations(img)    
# fEmbeddings = face_recognition.face_encodings(img,fLoc)[0]
# 
# result = face_recognition.compare_faces(list(knownFaces.values()), fEmbeddings,tolerance=0.5)
# 
# 
# model.predi
# =============================================================================








    
