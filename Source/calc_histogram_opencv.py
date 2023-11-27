import numpy as np
import cv2

def calc_histo(image, hsize, ranges=[0, 256]):
    hist = np.zeros((hsize, 1), np.float32)
    gap = ranges[1] / hsize

    for i in (image / gap).flat:
        hist[int(i)] += 1
    return hist

image = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)  # 영상읽기
if image is None:
    raise Exception("영상 파일 읽기 오류")

hsize, ranges = 32, [0, 256]  # 히스토그램 간격 수, 값 범위
gap = ranges[1] / hsize  # 계급 간격
ranges_gap = np.arange(0, ranges[1] + 1, gap)  # 넘파이 계급 범위&간격
hist1 = calc_histo(image, hsize, ranges)  # User 함수
hist2 = cv2.calcHist([image], [0], None, [hsize], ranges)  # OpenCV 함수
hist3, bins = np.histogram(image, ranges_gap)  # numpy 모듈 함수

print("User 함수: \n", hist1.flatten())  # 1차원 행렬 1행 표시
print("OpenCV 함수: \n", hist2.flatten())
print("numpy 함수: \n", hist3)

# 이미지를 새 창에 표시
cv2.imshow("Image", image)

# 키보드 입력을 기다림 (아무 키나 누를 때까지)
cv2.waitKey(0)

# 모든 창을 닫음
cv2.destroyAllWindows()
