import numpy as np
import cv2, math
import scipy.fftpack as sf

def cos(n, k, N): #코사인 함수
    return math.cos((n+1/2) * math.pi * k/N)

def C(k, N): #상수 계산 함수
    return math.sqrt(1/N) if k==0 else math.sqrt(2/N)

def dct(g): #1차원 dct 수행 함수
    N = len(g)
    f = [C(k, N) * sum(g[n] * cos(n, k, N) for n in range(N)) for k in range(N)]
    return np.array(f, np.float32)

def idct(F): #1차원 역dct 수행 함수
    N = len(F)
    g = [sum(C(k, N) * F[k] * cos(n, k, N) for k in range(N)) for n in range(N)]
    return np.array(g)

def dct2(image):
    tmp = [dct(row) for row in image]
    dst = [dct(row) for row in np.transpose(tmp)]
    return np.transpose(dst)

def idct2(image):
    tmp = [idct(row) for row in image]
    dst = [idct(row) for row in np.transpose(tmp)]
    return np.transpose(dst)

def scipy_dct2(a):
    tmp = sf.dct(a, axis=0, norm='ortho')
    return sf.dct(tmp, axis=1, norm='ortho')

def scipy_idct2(a):
    tmp = sf.idct(a, axis=0, norm='ortho')
    return sf.idct(tmp, axis=1, norm='ortho')
