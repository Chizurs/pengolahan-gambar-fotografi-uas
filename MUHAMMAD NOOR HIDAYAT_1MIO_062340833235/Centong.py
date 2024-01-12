import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 10, 10)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def remove_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])  
    upper_green = np.array([80, 255, 255])  
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

original_image = cv2.imread('Centong.jpg')

edge_image = edge_detection(original_image)

segmented_image = remove_background(original_image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Gambar Asli')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edge_image)
plt.title('Hasil Deteksi')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Hasil Segmentasi')
plt.axis('off')

plt.show()
