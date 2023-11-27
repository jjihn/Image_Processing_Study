import numpy as np
import cv2

src = cv2.imread('me_dark.jpg', cv2.IMREAD_GRAYSCALE)
if src is None:
    print('Image load failed!')
    sys.exit()

dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)
#히스토그램 스트레칭은 NORM_MINIMAX

#numpy로 히스토그램 스트레칭 구현
gmin = np.min(src)
gmax = np.max(src)
dst = np.clip((src - gmin) * 255. / (gmax - gmin), 0, 255).astype(np.uint8)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()
