# -*- coding: utf-8 -*-
import numpy as np
import cv2

img = cv2.imread('images/2.jpg')
rows,cols,ch = img.shape
img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR)
pts1 = np.float32([[640,0], [0,480], [0,0], [640,480]])
pts2 = np.float32([[187, 153], [422, 376], [315, 65], [306, 477]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(640,480))

cv2.imshow('title', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
