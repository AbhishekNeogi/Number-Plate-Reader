import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytesseract
import os,sys
import uuid
import pandas as pd

#importing image
img = cv.imread("Photos/image4.jpg")
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plt.show()

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
plt.show()

#applying some filter and edge detection
bifilter = cv.bilateralFilter(gray,11,17,17) #noise reduction
edged = cv.Canny(bifilter,30,200) #edge detectionq
plt.imshow(cv.cvtColor(edged,cv.COLOR_BGR2RGB))
plt.show()

#find contours and apply masking
points = cv.findContours(edged.copy(),cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(points)
contours = sorted(contours , key=cv.contourArea, reverse=True)[:10]

#locating coordinates of number plate rectangle
location = None
for contour in contours:
    approx = cv.approxPolyDP(contour,10,True)
    if len(approx)==4: #4 corners
        location = approx   #this means it is approximately a rectange
        break
print(location)

#extracting number plate from the whole image
mask = np.zeros(gray.shape, np.uint8)
new_img = cv.drawContours(mask , [location],0,255,-1)
new_img = cv.bitwise_and(img,img, mask=mask)

plt.imshow(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))
plt.show()

#use text recognization 
#interpreting the number plate,characters
(x,y) = np.where(mask == 255) #storing x and y values(coordinates)
x1 = np.min(x)
y1 = np.min(y)
x2 = np.max(x)
y2 = np.max(y)

cropped_img = gray[x1:x2+1,y1:y2+1]
plt.imshow(cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB))
plt.show()


fileid = f"{str(uuid.uuid4())}.png"
cv.imwrite(fileid,cropped_img)

nup =(pytesseract.image_to_string(fileid))
print(nup)

os.remove(fileid)
# print(type(nup))


# df = pd.DataFrame()



