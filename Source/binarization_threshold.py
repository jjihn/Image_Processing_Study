import numpy as np
import cv2
import matplotlib.pylab as plt

img = cv2.imread("me.jpg", cv2.IMREAD_GRAYSCALE)

# 1. NumPy API로 바이너리 이미지 만들기
thresh_np = np.zeros_like(img) #원본과 동일한 크기의 0으로 채워진 이미지
thresh_np[ img > 127] = 255 #127보다 큰 값만 255로 변경

# 2. OpenCV API로 바이너리 이미지 만들기
ret, thresh_cv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
print(ret) #127.0, 바이너리 이미지에 사용된 문턱 값 반환

# 3. 원본과 결과물을 matplotlib로 출력
imgs = {'Original': img, 'NumPy API': thresh_np, 'cv2.threshold': thresh_cv}
for i , (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()
