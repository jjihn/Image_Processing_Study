import numpy as np
import cv2

def bilinear_value(img, pt): #단일 화소 양선형 보간 수행 함수
    x,y = np.int32(pt)
    if x >= img.shape[1]-1: x=x-1 #영상 범위 벗어남 처리
    if y >= img.shape[0]-1: y=y-1

    P1, P2, P3, P4 = np.float32(img[y:y+2, x:x+2].flatten()) #4개 화소 - 관심영역으로 접근

    alpha, beta = pt[1] - y, pt[0] - x #거리 비율
    M1 = P1 + alpha * (P3 - P1) #1차 보간
    M2 = P2 + alpha * (P4 - P2) 
    P = M1 + beta * (M2 - M1) #2차 보간
    return np.clip(P, 0, 255) #화소값 saturation 후 반환
