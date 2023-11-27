import numpy as np
import cv2
from Common.dct2d import dct2, idct2, scipy_dct2, scipy_idct2

def dct2_mode(block, mode):
    if mode == 1: return dct2(block)
    elif mode == 2: return scipy_dct2(block)
    elif mode == 3: return cv2.dct(block.astype('float32'))

def idct2_mode(block, mode):
    if mode == 1: return idct2(block)
    elif mode == 2: return scipy_idct2(block)
    elif mode == 3: return cv2.dct(block, flags=cv2.DCT_INVERSE)

def dct_filtering(img, filter, M, N): #주파수 영역 필터링 함수
    dst = np.empty(img.shape, np.float32)
    for i in range(0, img.shape[0], M): #입력 영상 순회
        for j in range(0, img.shape[1], N):
            block = img[i:i+M, j:j+N] #블록 참조
            new_block = dct2_mode(block, mode) #블록 DCT 수행
            new_block = new_block * filter #곱셈을 통한 필터링
            dst[i:i+M, j:j+N] = idct2_mode(new_block, mode) #역DCT
    return cv2.convertScaleAbs(dst)

image = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("Image load failed!")

mode = 3 #dct 방식 선택
M, N = 8, 8 #블록 크기
filters = [np.zeros((M,N), np.float32) for i in range(5)] #5개 필터 생성, 0으로 초기화
titles = ['DC Pass', 'High Pass', 'Low Pass', 'Vertical Pass', 'Horizental Pass']

filters[0][0, 0] = 1 #DC 계수만 1 지정 - DC pass
filters[1][:], filters[1][0,0] = 1,0 #모든 계수 1, DC만 0 - High pass
filters[2][:M//2, :N//2] = 1 # 1/4영역 1 지정 -Low pass
filters[3][0, 1:] = 1 #수직 성분 -Vertical pass
filters[4][1:, 0] = 1 #수평 성분 -Horizental pass

for filter, title in zip(filters, titles):
    dst = dct_filtering(image, filter, M, N)
    cv2.imshow(title, dst)
cv2.imshow("image", image)
cv2.waitKey(0)


            
