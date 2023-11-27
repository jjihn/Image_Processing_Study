import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("me.jpg")

# 소벨 커널을 직접 생성해서 엣지 검출
gx_k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # 소벨 커널 생성
gy_k = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
edge_gx = cv2.filter2D(img, -1, gx_k)  # 소벨 필터 적용
edge_gy = cv2.filter2D(img, -1, gy_k)

# 소벨 API를 사용하여 엣지 검출
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)

# 이미지를 서브플롯으로 표시
plt.figure(figsize=(10, 5))

# 원본 이미지와 소벨 커널로 엣지 검출 결과
plt.subplot(2, 4, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 4, 2)
plt.title("Edge Detection (GX)")
plt.imshow(cv2.cvtColor(edge_gx, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 4, 3)
plt.title("Edge Detection (GY)")
plt.imshow(cv2.cvtColor(edge_gy, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 4, 4)
plt.title("Edge Detection (GX+GY)")
plt.imshow(cv2.cvtColor(edge_gx + edge_gy, cv2.COLOR_BGR2RGB))
plt.axis("off")

# 원본 이미지와 소벨 API로 엣지 검출 결과
plt.subplot(2, 4, 5)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 4, 6)
plt.title("Edge Detection (SobelX)")
plt.imshow(cv2.cvtColor(sobelx, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 4, 7)
plt.title("Edge Detection (SobelY)")
plt.imshow(cv2.cvtColor(sobely, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 4, 8)
plt.title("Edge Detection (SobelX+SobelY)")
plt.imshow(cv2.cvtColor(sobelx + sobely, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
