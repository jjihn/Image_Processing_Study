import cv2
import numpy as np

title = 'Subway Perspective'                   # 창 제목
img_base = cv2.imread('subway.jpg')        #  이미지 읽기
rows, cols = img_base.shape[:2]

cv2.imshow(title, img_base)              


img_fg = cv2.imread('me.jpg')
rows, cols = img_fg.shape[:2]

#---① 원근 변환 전 후 4개 좌표
pts1 = np.float32([[0,0], [0,rows], [cols, 0], [cols,rows]])
pts2 = np.float32([[94,113], [94,321], [497, 33], [494,425]])

#---② 변환 전 좌표를 원본 이미지에 표시
cv2.circle(img_fg, (0,0), 10, (255,0,0), -1)
cv2.circle(img_fg, (0,rows), 10, (0,255,0), -1)
cv2.circle(img_fg, (cols,0), 10, (0,0,255), -1)
cv2.circle(img_fg, (cols,rows), 10, (0,255,255), -1)

#---③ 원근 변환 행렬 계산
mtrx = cv2.getPerspectiveTransform(pts1, pts2)
#---④ 원근 변환 적용
dst = cv2.warpPerspective(img_fg, mtrx, (cols, rows), cv2.BORDER_CONSTANT,0)

cv2.imshow("origin", img_fg)
cv2.imshow('perspective', dst)

#--- dst를 Gray영상으로 변환 dst2 하고 threshold하여 mask를 만든다.
dst2=cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(dst2, 0, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)

mask_inv = mask_inv.astype(np.uint8)
if mask_inv.shape[:2] != img_base.shape[:2]:
    mask_inv = cv2.resize(mask_inv, (img_base.shape[1], img_base.shape[0]))

#---mask_inv와 배경영상을 AND하고 그 결과를 원근변환 영상과 더한다.
masked_base = cv2.bitwise_and(img_base, img_base, mask=mask_inv)
cv2.imshow('masked_base', masked_base)

img_added = cv2.add(masked_base, dst)
#img_added = masked_base+dst
cv2.imshow('img_added', img_added)

#cv2.imwrite('billboard_img.jpg', img_added)

cv2.waitKey(0)
cv2.destroyAllWindows()
