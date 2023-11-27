import numpy as np
import cv2
import math
from Common.hough import accumulate, masking, select_line #허프 변환 함수 임포트

def houghLines(src, rho, theta, thresh): #허프 변환 함수
    acc_mat = accumulate(src, rho, theta) #직선 누적 행렬 계산
    acc_dst = masking(acc_mat, 7, 3, thresh) #마스킹 처리 - 7행, 3열
    lines = select_line(acc_dst, rho, theta, thresh) #임계 직선 선택
    return lines

def draw_houghLines(src, lines, nline): #검출 직선 그리기 함수
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR) #컬러 영상 변환
    min_length = min(len(lines), nline)

    for i in range(min_length):
        rho, radian = lines[i, 0, 0:2] #수직 거리, 각도 - 3차원 행렬임
        a, b = math.cos(radian), math.sin(radian)
        pt = (a * rho, b * rho)
        delta = (-1000 * b, 1000 * a)
        pt1 = np.add(pt, delta).astype('int')
        pt2 = np.subtract(pt, delta).astype('int')
        cv2.line(dst, tuple(pt1), tuple(pt2), (0, 255, 0), 2, cv2.LINE_AA)

    return dst

image = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("Image load failed!")
blur = cv2.GaussianBlur(image, (5,5), 2, 2) #가우시안 블러링
canny = cv2.Canny(blur, 100, 200, 5) #캐니 엣지 추출

rho, theta = 1, np.pi / 180 #수직 거리 간격, 각도 간격
lines1 = houghLines(canny, rho, theta, 80)
lines2 = cv2.HoughLines(canny, rho, theta, 80)
dst1 = draw_houghLines(canny, lines1, 7) #직선 그리기
dst2 = draw_houghLines(canny, lines2, 7)

cv2.imshow("image", image)
cv2.imshow("canny", canny)
cv2.imshow("detected lines", dst1)
cv2.imshow("detected lines_OpenCV", dst2)
cv2.waitKey(0)
