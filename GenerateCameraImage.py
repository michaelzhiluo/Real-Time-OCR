import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while (True) :
	# Capture frame - by - frame
	ret, frame = cap.read()


	if cv2.waitKey(1) & 0xFF == ord('q') :
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()