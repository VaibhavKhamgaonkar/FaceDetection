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
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import FaceTraining as ft


class unknownFacesTraining():
    
    def __init__(self):
        self.path = os.path.dirname(os.path.abspath('__file__')) + '/'
        self.faceDBPath = self.path + 'FacesDB/'
    
        #grabbing the unknown faces first
        
        self.unkownFdr = [folder for folder in os.listdir(self.faceDBPath) if 'Unknown' in folder][0]
        
        self.images = os.listdir(self.faceDBPath+self.unkownFdr)
        self.images = unknownFacesTraining.sort(self.images)
        self.embeddings = []
        self.loc = []
        
    
    # =============================================================================
    # defining the customised function for sorting the data
    # =============================================================================
    def sort(images):
        images = [int(item.split('.')[0]) for item in images]
        images.sort()
        return [str(item)+'.jpg' for item in images]
        
     
    # =============================================================================
    # Function to copy the file from source to destination    
    # =============================================================================
    
    def copyFile(srcPath, dstPath):
        shutil.copy2(srcPath,dstPath)
    
    
    
    def playWithUnknownImages(self):
        
        for image in self.images :
            
            img = cv2.imread(self.faceDBPath+self.unkownFdr + '/' + image)
            
            try:
                faceLoc = face_recognition.face_locations(img)
                faceEncodings = face_recognition.face_encodings(img,faceLoc)[0]
                
            except Exception as e:
                #print(e)
                #removing Junk images
                os.remove(self.faceDBPath+self.unkownFdr + '/' + image)
                continue
            self.embeddings.append(faceEncodings)
            self.loc.append(self.faceDBPath + self.unkownFdr + '/' + image)
            
        
        # =============================================================================
        # gmm = GaussianMixture(n_components=0)  
        # 
        # gmm.fit(np.array(embeddings))
        # 
        # =============================================================================
        inertia = []
        for k in range(2,10):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.embeddings)
            print(kmeans.inertia_)
            inertia.append(kmeans.inertia_)
        
        plt.plot(inertia)
        plt.title("Identify the correct Elbow Point and note the value..")
        plt.xlabel('Cluster group')
        plt.ylabel('Value')
        plt.show()
           
        #from the graph choosing the best value of cluster
        k =  int(input('Based on the plot, please enter the number to divide the unknown images into group.\n', ))
        
        kmeans = KMeans(n_clusters=k+1)
        kmeans.fit(self.embeddings)
        
        #getting the labels
        
        cluster = kmeans.labels_
        #creating a data frame of all the details
        #data = pd.DataFrame([cluster,self.loc, self.embeddings]).transpose()
        data = pd.DataFrame([cluster,self.loc]).transpose()
        
        # =============================================================================
        # Asking user to update the details for unkown person
        # =============================================================================
        
        uniqueImgs = data.groupby(by=0).first() # getting record of each unique person
        name = []
        for i, item in enumerate(uniqueImgs[1]): # looping through it to get the name of unkow person
            
            pic = cv2.imread(item)
            cv2.putText(pic, 'Do you want to add this person to Database ? Press "q" to close this window', (0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
            
            cv2.imshow('Image',pic)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            choice = str(input("do you want to add record to Data set ? y/n ? \n", ))
            if choice.lower() == 'y':
                name.append(str(input(' Enter the name for this Record. \n', )))
            else:
                name.append(None)
                data.drop(data[data[0]==i].index)
                
        cv2.destroyAllWindows()
        #creating a label from this details and updating the Data frame
        data['Name'] = data[0].apply(lambda x : str(name[x]))
            
        
        #Distributing the images to the respective folders and then training the data set again
        
        for nme in name:
            if nme != None:
                #check the directory exists or not
                if not os.path.exists(self.faceDBPath + str(nme)):
                        os.mkdir(self.faceDBPath + str(nme))
                        
                #getting the index of the name
                index = data[data['Name'] == nme].index
                for idx in  index:
                    unknownFacesTraining.copyFile(data[1].iloc[idx], self.faceDBPath + str(nme) + '/')
                
                print('Files copied to ', nme)
        
        
        
        
        # =============================================================================
        # Now Traning the Model
        # =============================================================================
        tr = ft.FaceTraining()
        tr.train()
