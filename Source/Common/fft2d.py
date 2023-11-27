import numpy as np
import cv2

def zeropadding(img):
    h, w = img.shape[:2]
    m = 1 << int(np.ceil(np.log2(h))) #2의 자승 계산
    n = 1 << int(np.ceil(np.log2(w)))
    dst = np.zeros((m, n), img.dtype) #2의 자승 크기 영상 생성
    dst[0:h, 0:w] = img[:] #자승 크기 영상에 원본 영상 복사
    return dst

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

def butterfly(pair, L, N, dir):
    for k in range(L):
        Geven, Godd = pair[k], pair[k+L]
        pair[k] = Geven + Godd * exp(dir * k / N)
        pair[k+L] = Geven - Godd * exp(dir * k / N)

def parring(g, N, dir, start=0, stride=1):
    if N == 1: return [g[start]]
    L = N // 2
    sd = stride *2
    part1 = parring(g, L, dir, start, sd)
    part2 = parring(g, L, dir, start + stride, sd)
    pair = pair1+pair2
    butterfly(pair, L, N, dir)
    return pair

def fft(g):
    return pairing(g, len(g), 1)

def ifft(g):
    fft = pairing(g, len(g), -1)
    return [v/len(g) for v in fft]

def fft2(image):
    pad_img = zeropadding(image)
    tmp = [fft(row) for row in pad_img]
    dst = [fft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)

def ifft2(image):
    tmp = [ifft(row) for row in image]
    dst = [ifft(row) for row in np.transpose(timp)]
    return np.transpose(dst)
    
