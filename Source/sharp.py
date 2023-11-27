import numpy as np
import cv2

img = cv2.imread("me.jpg")

#이미지 필터링 커널 정의
kernel_sharpen_1 = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

kernel_sharpen_2 = np.array([[-1,-1,-1],
                             [-1, 9, -1],
                             [-1, -1, -1]])

kernel_sharpen_3 = np.array([[1, 1, 1],
                             [1, -7, 1],
                             [1, 1, 1]])

kernel_sharpen_4 = np.array([[-1, -1, -1, -1, -1],
                            [-1,  2,  2,  2, -1],
                            [-1,  2,  8,  2, -1],
                            [-1,  2,  2,  2, -1],
                            [-1, -1, -1, -1, -1]], dtype=np.float32) / 8.0

#이미지 필터링 적용
sharp_image_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
sharp_image_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
sharp_image_3 = cv2.filter2D(img, -1, kernel_sharpen_3)
sharp_image_4 = cv2.filter2D(img, -1, kernel_sharpen_4)
                             
cv2.imshow("Original Image", img)
cv2.imshow("Sharpened Image 1", sharp_image_1)
cv2.imshow("Sharpened Image 2", sharp_image_2)
cv2.imshow("Sharpened Image 3", sharp_image_3)
cv2.imshow("Sharpened Image 4", sharp_image_4)

cv2.waitKey(0)
cv2.destroyAllWindow()
