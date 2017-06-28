import numpy as np
import cv2

counter =0
roi = []
# 0 -> x1, 1 -> x2, 2 -> y1, 3-> y2
roi += [[320, 345, 115, 175]]
roi += [[345, 373, 115, 175]]
roi += [[372, 399, 115, 175]]
roi += [[398, 427, 115, 175]]
roi += [[438, 465, 115, 175]]
roi += [[465, 491, 115, 175]]
roi += [[268, 292, 305, 365]]
roi += [[292, 318, 305, 365]]
roi += [[319, 344, 305, 365]]
roi += [[345, 368, 305, 365]]
roi += [[370, 395, 305, 365]]
roi += [[396, 423, 305, 365]]
roi += [[435, 462, 305, 365]]
roi += [[463, 487, 305, 365]]
numpics=0
while(counter <=20000):
	frame =  cv2.imread("C:\\Users\\Michael Luo\\Documents\\ComputerDigitsTrainingImages\\" + str(counter) + ".jpg");  
	for i in range(0, len(roi)):
		temp = frame[roi[i][2]: roi[i][3], roi[i][0]: roi[i][1]]
		cv2.imshow("roi " + str(i), temp)
		cv2.imwrite("C:\\Users\\Michael Luo\\Documents\\TrainingData\\" + str(numpics) + ".jpg", temp)
		numpics+=1
	for i in range(0, len(roi)):
		cv2.rectangle(frame, (roi[i][0], roi[i][2]), (roi[i][1], roi[i][3]), (255, 255, 255))
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	counter+=1


cv2.destroyAllWindows()