import numpy as np
import cv2
import matplotlib.pyplot as plt

file_name = "me.jpg"
img = cv2.imread(file_name)

# 프리윗 커널 생성
gx_k = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
gy_k = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# 프리윗 커널 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 이미지를 서브플롯으로 표시
plt.figure(figsize=(10, 5))

# 원본 이미지
plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

# 엣지 감지 결과
plt.subplot(1, 4, 2)
plt.title("Edge Detection (GX)")
plt.imshow(cv2.cvtColor(edge_gx, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Edge Detection (GY)")
plt.imshow(cv2.cvtColor(edge_gy, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Edge Detection (GX+GY)")
plt.imshow(cv2.cvtColor(edge_gx + edge_gy, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
