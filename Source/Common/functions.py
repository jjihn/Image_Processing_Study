import numpy as np
import cv2

def contain(p,shape): #좌표(y,x)가 범위 내 인지 검사
    return 0<= p[0] <shape[0] and 0 <= p[1] < shape[1] 
