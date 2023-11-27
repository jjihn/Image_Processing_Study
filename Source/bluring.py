import numpy as np
import cv2

#회선 수행 함수 - 행렬 처리 방
def filter(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32) #회선 결과 저장 행렬
    ycenter, xcenter = rows//2, cols//2 #마스크 중심 좌표

    for i in range(ycenter, rows - ycenter): #입력 행렬 반복 순회
        for j in range(xcenter, cols - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1 #관심 영역 높이 범위
            x1, x2 = j = xcenter, j + xcenter + 1 #관심 영역 너비 범위
            roi = image[y1:y2, x1:x2].astype('float32') #관심 영역 형변환
            tmp = cv2.multiply(roi, mask) # 회선 적용 - 원소 간 곱셈
            dst[i, j] = cv2.sumElems(tmp)[0] # 출력 화소 저장
    return dst # 자료형 변환하여 반환

#회선 수행 함수 - 화소 직접 근접
def filter2(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows,cols), np.float32)
    ycenter, xcenter = rows//2, cols//2

    for i in range(ycenter, rows - ycenter):
        for j in range(xcenter, cols - xcenter):
            sum = 0.0
            for u in range(mask.shape[0]): #마스크 원소 순회
                for v in range(mask.shape[1]):
                    y,x = i + u - ycenter, j + v - xcenter
                    sum += image[y, x] * mask[u, v] #회선 수식
            dst[i,j] = sum
    return dst

image = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("Image load failed!")

data = np.array([[1/9, 1/9, 1/9],  #블러링 마스크 원소 지정
                [1/9, 1/9, 1/9],
                [1/9, 1/9, 1/9]])
mask = np.array(data, np.float32).reshape(3, 3) #마스크 행렬 생성
blur1 = filter(image, mask) #회선 수행 - 행렬 처리 방식
blur2 = filter2(image, mask) #회선 수행 - 화소 직접 접근
blur1 = blur1.astype('uint8') #행렬 표시 위해 uint8형 변환
blur2 = cv2.convertScaleAbs(blur2)

cv2.imshow("image", image)
cv2.imshow("blur1", blur1)
cv2.imshow("blur2", blur2)
cv2.waitKey(0)
