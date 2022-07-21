import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
#img = cv2.imread('rdj.webp')
#img = cv2.imread('mnd.jpg')

#to capture video from webcam
webcam = cv2.VideoCapture(0)

#iterate forever over frames
while True:
         #read the current frame
         successful_frame_read, frame = webcam.read()

         # must convert to grayscale
         grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


         # detect faces
         face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

         #draw rectangles around the faces 
         #for (x,y,w,h) in face_coordinates:(x,y),(x+w,y+h),(0,255,0)=> (B,G,R), 2=> thickness
        #cv2.rectangle(img,(192,183),(192+404,183+404),(0,255,0),2)
         for (x,y,w,h) in face_coordinates:
           cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(126,256),randrange(126,256),randrange(126,256)),10)

         # Shows the image
         cv2.imshow('face detector',frame)

         # hold the image until a key is pressed
         cv2.waitKey(1)



print("code completed")