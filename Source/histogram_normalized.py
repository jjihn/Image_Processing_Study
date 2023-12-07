import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# 히스토그램 계산
hist, bins = np.histogram(image.flatten(), 256, [0,256])

# 누적 히스토그램 계산
cdf = hist.cumsum()

# 히스토그램 정규화
cdf_normalized = cdf * hist.max() / cdf.max()

# 정규화된 히스토그램의 값으로 원본 이미지의 픽셀 값 변경
image_normalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)

# 이미지 차원 변경
image_normalized = image_normalized.reshape(image.shape)

# 결과 히스토그램 계산
hist_normalized, bins_normalized = np.histogram(image_normalized.flatten(), 256, [0,256])

# 결과 시각화
plt.subplot(121), plt.imshow(image, 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(image_normalized, 'gray')
plt.title('Normalized Image'), plt.xticks([]), plt.yticks([])
plt.show()
