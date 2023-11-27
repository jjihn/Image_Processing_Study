import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

def exp(knN):
    th = -2 * math.pi * knN #푸리에 변환 각도 값
    return complex(math.cos(th), math.sin(th)) #복소수 클래스

def dft(g):
    N = len(g)
    dst = [sum(g[n] * exp(k*n/N) for n in range(N)) for k in range(N) ]
    return np.array(dst)

def idft(G):
    N = len(G)
    dst = [sum(G[n] * exp(-k*n/N) for n in range(N)) for k in range(N) ]
    return np.array(dst) / N

def calc_spectrum(complex):
    if complex.ndim==2:
        dst = abs(complex) #sqrt(re^2+im^2) 계산
    else:
        dst = cv2.magnitude(complex[:, :, 0], complex[:, :, 1]) #OpenCV의 경우
    dst = cv2.log(dst+1)
    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(dst)

def fftshift(img):
    dst = np.zeros(img.shape, img.dtype)
    h, w = dst.shape[:2]
    cy, cx = h // 2, w // 2 #몫 연산자
    dst[h-cy:, w-cx:] = np.copy(img[0:cy, 0:cx]) #1사분면 -> 3사분면
    dst[0:cy, 0:cx] = np.copy(img[h-cy:, w-cx:]) #3사분면 -> 1사분면
    dst[0:cy, w-cx:] = np.copy(img[h-cy:, 0:cx]) #2사분면 -> 4사분면
    dst[h-cy:, 0:cx] = np.copy(img[0:cy, w-cx:]) #4사분면 -> 2사분면
    return dst

def zeropadding(img):
    h, w = img.shape[:2]
    m = 1 << int(np.ceil(np.log2(h))) #2의 자승 계산
    n = 1 << int(np.ceil(np.log2(w)))
    dst = np.zeros((m, n), img.dtype) #2의 자승 크기 영상 생성
    dst[0:h, 0:w] = img[:] #자승 크기 영상에 원본 영상 복사
    return dst
