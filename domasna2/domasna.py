import cv2
import numpy as np

# Вчитување на сликата
img = cv2.imread('Screenshot from 2024-04-13 19-17-46.png', cv2.IMREAD_GRAYSCALE)

# Дефинирање на Compass операторот
compass_operator = np.array([
    [ 1,  1, -1, -1, -1,  1,  1,  1],
    [ 1,  1,  1, -1, -1, -1,  1,  1],
    [ 1,  1,  1,  1, -1, -1, -1,  1],
    [ 1,  1,  1,  1,  1, -1, -1, -1],
    [-1,  1,  1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1,  1,  1, -1],
    [-1, -1, -1,  1,  1,  1,  1,  1],
    [-1, -1, -1, -1,  1,  1,  1,  1]
])

# Примена на Compass операторот на сликата
edges = np.zeros_like(img)
for i in range(8):
    kernel = np.rot90(compass_operator, i)
    edges = np.maximum(edges, cv2.filter2D(img, -1, kernel))

# Приказ на резултатот
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, thresholded_edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

# Приказ на резултатот
cv2.imshow('Thresholded Edges', thresholded_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
