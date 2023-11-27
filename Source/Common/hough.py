import numpy as np
import cv2

def accumulate(image, rho, theta):
    h, w = image.shape[:2]
    rows, cols, = (h+w) *2 // rho, int(np.pi / theta) #누적행렬 너비, 높이
    accumulate = np.zeros((rows,cols), np.int32) #직선 누적행렬

    sin_cos = [(np.sin(t*theta), np.cos(t*theta)) for t in range(cols)] #삼각함수 값 저장
    pts = np.where(image > 0) #numpy함수 활용 - 직선 좌표 찾기

    polars = np.dot(sin_cos, pts).T #행렬 곱으로 극좌표 계산
    polars = (polars / rho+rows / 2).astype('int') #해상도 변경 및 위치 조정
    for row in polars:
        for t, r in enumerate(row): #각도, 수직 거리 가져옴
            accumulate[r, t] += 1 #극좌표에 누적
    return accumulate

def masking(accumulate, h, w, thresh):
    rows, cols = accumulate.shape[:2]
    rcenter, tcenter = h//2, w//2 #마스크 크기 절반
    dst = np.zeros(accumulate.shape, np.uint32)

    for y in range(0, rows, h):
        for x in range(0, cols, w):
            roi = accumulate[y:y+h, x:x+w]
            _, max, _, (x0, y0) = cv2.minMaxLoc(roi)
            dst[y+y0, x+x0] = max
    return dst

def select_line(acc_dst, rho, theta, thresh):
    rows = acc_dst.shape[0]
    r, t = np.where(acc_dst>thresh) #임계값 이상 인덱스 가져옴

    rhos = ((r-(rows / 2))*rho) #인덱스로 수직 거리 계산
    radians = t*theta #인덱스로 각도 계산
    values = acc_dst[r,t] #인덱스로 누적값 가져옴

    idx = np.argsort(values)[::-1] #내림차순 정렬 인덱스
    lines = np.transpose([rhos, radians]) #리스트 전치하여 행렬 생성
    lines = lines[idx, :] #누적값에 다른 정렬

    return np.expand_dims(lines, axis=1) #1번 열 차원 증가
    
