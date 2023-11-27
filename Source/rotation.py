import numpy as np
import cv2
import math
from Common.interpolation import bilinear_value
from Common.functions import contain

def rotate(img, degree): #원점 기준 회전 변환 함수
    dst = np.zeros(img.shape[:2], img.dtype) #목적 영상 생성
    radian = (degree/180) * np.pi #회전 각도-라디안
    sin, cos = np.sin(radian), np.cos(radian) #사인, 코사인 값 계산

    for i in range(img.shape[0]): #목적 영상 순회 - 역방향 사상
        for j in range(img.shape[1]): 
            y = -j * sin + i * cos
            x = j * cos + i * sin #회선 변환 수식
            if contain((y,x), img.shape): #입력 영상의 범위 확인
                dst[i,j] = bilinear_value(img, [x,y]) #화소값 양선형 보간
    return dst

def rotate_pt(img, degree, pt): #pt 기준 회전 변환 함수
    dst = np.zeros(img.shape[:2], img.dtype) #목적 영상 생성
    radian = (degree/180) * np.pi #회전 각도-라디안
    sin, cos = math.sin(radian), math.cos(radian) #사인 코사인 계산

    for i in range(img.shape[0]): #목적 영상 순회 - 역방향 사상
        for j in range(img.shape[1]):
            jj, ii = np.subtract((j, i), pt) #중심 좌표로 평행 이동
            y = -jj * sin + ii * cos #회선 변환 수식
            x = jj * cos + ii *sin
            x, y = np.add((x,y),pt) #중심 좌표로 평행이동
            if contain((y,x), img.shape): #입력 영상의 범위 확인
                dst[i,j] = bilinear_value(img,(x,y)) #양선형 보간
    return dst

image = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("Image load failed!")

center = np.divmod(image.shape[::-1], 2)[0] #영상 크기로 중심 좌표 계산
dst1 = rotate(image, 20) #원점 기준 회전 변환
dst2 = rotate_pt(image, 20, center) #center기준 회전 변환

cv2.imshow("Image", image)
cv2.imshow("dst1 : rotated on (0,0)", dst1);
cv2.imshow("dst2 : rotated on center point", dst2)
cv2.waitKey(0)
