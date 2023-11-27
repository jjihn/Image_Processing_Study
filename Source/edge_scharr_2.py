import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("me.jpg")

# 샤르 커널을 직접 생성해 엣지 검출
gx_k = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
gy_k = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 샤르 API로 엣지 검출
scharrx = cv2.Scharr(img, -1, 1, 0)
scharry = cv2.Scharr(img, -1, 0, 1)

# 이미지를 서브플롯으로 표시
plt.figure(figsize=(10, 5))

# 원본 이미지와 직접 생성한 샤르 커널로 엣지 검출 결과
plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Edge Detection (GX)")
plt.imshow(cv2.cvtColor(edge_gx, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Edge Detection (GY)")
plt.imshow(cv2.cvtColor(edge_gy, cv2.COLOR_BGR2RGB))
plt.axis("off")

# 원본 이미지와 샤르 API로 엣지 검출 결과
plt.subplot(2, 3, 4)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("Edge Detection (ScharrX)")
plt.imshow(cv2.cvtColor(scharrx, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 6)
plt.title("Edge Detection (ScharrY)")
plt.imshow(cv2.cvtColor(scharry, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
