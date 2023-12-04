import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Image load failed!!")
        return

    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    return histogram

def plot_histogram(histogram):
    plt.plot(histogram, color='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Image Histogram')
    plt.grid()
    plt.show()

image_path = 'me.jpg'  # 이미지 경로를 지정해주세요
histogram = calculate_histogram(image_path)
plot_histogram(histogram)
print(histogram)
