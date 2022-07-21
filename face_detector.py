import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
#img = cv2.imread('rdj.webp')
img = cv2.imread('mnd.jpg')

# must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#draw rectangles around the faces 
#for (x,y,w,h) in face_coordinates:(x,y),(x+w,y+h),(0,255,0)=> (B,G,R), 2=> thickness
#cv2.rectangle(img,(192,183),(192+404,183+404),(0,255,0),2)
for (x,y,w,h) in face_coordinates:
 cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(126,256),randrange(126,256),randrange(126,256)),2)

#to print face co ordinates
#print(face_coordinates) 


# Shows the image
cv2.imshow('face detector',img)

# hold the image until a key is pressed
cv2.waitKey()


print("code completed")