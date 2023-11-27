import numpy as np
import cv2

def getGrayHistImage(hist):   
    imgHist = np.full((100, 256), 255, dtype=np.uint8)      # 가장 높은 높이가 100으로 제한을 둠
    histMax = np.max(hist)                               # histmax = 255
    for x in range(256):
        pt1 = (x, 100)                                   # 시작점, 좌측 상단 기준
        pt2 = (x, 100 - int(hist[x, 0] * 100 / histMax))     # 끝점, 100을 곱하고 255로 나눠 단위 통일
        cv2.line(imgHist, pt1, pt2, 0)                      # 직선을 그려 히스토그램 그리기
    return imgHist

src = cv2.imread('me_bright.jpg', cv2.IMREAD_GRAYSCALE)
if src is None:
    print('Image load failed!')
    sys.exit()
    
hist = cv2.calcHist([src], [0], None, [256], [0, 256])
histImg = getGrayHistImage(hist)
 
cv2.imshow('src', src)
cv2.imshow('histImg', histImg)
cv2.waitKey()
cv2.destroyAllWindows()
