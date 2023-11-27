import numpy as np
import cv2
import time
from Common.dft2d import dft, idft, calc_spectrum, fftshift

def dft2(image):
    tmp = [dft(row) for row in image]
    dst = [dft(row) for row in np.transpose(tmp)]
    return np.transpose(dst) #전치 환원 후 반환

def idft2(image):
    tmp = [idft(row) for row in image]
    dst = [idft(row) for row in np.transpose(tmp)]
    return np.transpose(dst) #전치 환원 후 반환

def ck_time(mode=0):
    global stime #수행시간 체크 함수
    if (mode ==0 ): #함수 내부에서 값 유지위해서
        stime = time.perf_counter()
    elif (mode==1):
        etime = time.perf_counter()
        print("수행시간 = %.5f sec" % (etime-stime)) #초 단위 경과 시간

image = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("Image load failed!")

ck_time(0) #시작 시간 체크
dft = dft2(image) #2차원 DFT 수행
spectrum1 = calc_spectrum(dft) #주파수 스펙트럼 영상
spectrum2 = fftshift(spectrum1) #np.fft.fftshift() 사용가능
idft = idft2(dft).real #2차원 IDFT 수행
ck_time (1) #종료시간 체크

cv2.imshow("Image", image)
cv2.imshow("Spectrum1", spectrum1)
cv2.imshow("Spectrum2", spectrum2)
cv2.imshow("idft_img", cv2.convertScaleAbs(idft))
cv2.waitKey(0)
