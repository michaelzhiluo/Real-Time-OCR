
import cv2
import numpy as np

originalimg = cv2.imread("Three.jpg")
gray = cv2.cvtColor(originalimg, cv2.COLOR_BGR2GRAY)
resize = cv2.resize(gray, dsize =(28, 28))
blur = cv2.GaussianBlur(resize,(5,5),8)
cv2.imshow("resize", blur)
cv2.waitKey(1000)
nump = np.asarray(resize, np.float32)
nump = (nump -np.min(nump))/(np.max(nump) - np.min(nump))