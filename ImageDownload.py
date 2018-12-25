from google_images_download import google_images_download
import os, numpy as np, time
import cv2
import dlib
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-p','--peopleList',required=True, help="list of people whose image to be downlaoded from google. It should be in the fom of python list")
ap.add_argument('-l','--limit',required=True,help="Limits for the image download")

args = vars(ap.parse_args())

baseDir =os.path.dirname(os.path.abspath('__file__')) + '/'
wPath =   baseDir + 'FacesDB/'

#faceDetect = cv2.CascadeClassifier( baseDir + '/haarcascades/haarcascade_frontalface_alt2.xml')
faceDetector = dlib.get_frontal_face_detector()
faceModel_path = 'shape_predictor_68_face_landmarks.dat'

landMarker = dlib.shape_predictor(faceModel_path)
#Initiate the Download Object
g = google_images_download.googleimagesdownload()

#germany footbal team players
"""players = ['Jerome Boateng','Matthias Ginter', 'Jonas Hector', 'Mats Hummels', 'Joshua Kimmich', 'Marvin Plattenhardt', 'Antonio Ruediger','Niklas Suele', 'Jonas Hector',
           'Sami Khedira','Julian Draxler','Toni Kroos', 'Thomas Müller','Leon Goretzka',
           'Joshua Kimmich', 'Sebastian Rudy', 'Julian Brandt', 'Ilkay Gündogan', 'Timo Werner', 'Mesut Ozil',
           'Marco Reus','Mario Gómez']
"""
if ',' in str(args['peopleList']):
    players = str(args['peopleList']).split(',')  
else:
    players = [str(args['peopleList'])]
print(players)


# =============================================================================
# Process the frontal faceof the images
# =============================================================================   
def processFrontFace(dirName):
    images = os.listdir(wPath + dirName)
    images = [image for image in images if '.DS_Store' not in image]
    
    for k, image in enumerate (images):
        img = cv2.imread(wPath + dirName + '/' + image)
        try:
            faces = faceDetector(img,0)
        except:
            continue
        #faces = faceDetect.detectMultiScale(img,scaleFactor=1.5,minNeighbors=5)
        if len(faces) > 0:
            for i in range (0,len(faces)):
                newRect = dlib.rectangle(int(faces[i].left() ),
                                int(faces[i].top() ),
                                int(faces[i].right() ),
                                int(faces[i].bottom() )
                            )
      
                #Getting x,y,w,h coordinate
                x = newRect.left()
                y = newRect.top()
                w = newRect.right() - x
                h = newRect.bottom() - y
                
                roi_color = img[y:y+h,x:x+w]
                
                #roi_gray = gray[y:y+h,x:x+w]
                #store the image 
                try:
                    roi_color = cv2.resize(roi_color, (300,300), cv2.INTER_AREA)
                    print(dirName + image)
                    cv2.imwrite(wPath + dirName + '/' +  str(image) , roi_color) #str(k) + '.jpg'
                except Exception as e:
                    #print(wPath + dirName + '/' +  str(image))
                    continue
                if k % 50 == 0:
                    print('{} images saved so far..'.format(k))

    return 'Done with {}'.format(dirName)



# =============================================================================
# Download the Images to specific Directory
# =============================================================================
def download(name,limit):
    #dirName= str(name)
    #if not os.path.exists(wPath + dirName):
        #os.makedirs(wPath + dirName)
    #you will require chrome driver if images are more than 100 mention the path here     
    arguments = {"keywords":str(name),"limit":limit,"print_urls":True,'type':'face',
                 'format':'jpg',
                 "output_directory" : wPath, 'chromedriver':'/Users/vk250027/ChromeDriver/chromedriver'  } 
    g.download(arguments)
    print(processFrontFace(str(name)))
    

    
#looping the player list and download the images for each players
for player in players:
    print('Starting image download for: ', player )
    download(player, limit=args['limit'])



