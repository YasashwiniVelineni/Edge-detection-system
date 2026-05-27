import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('/workspaces/Edge-detection-system/input.jpg', 0)

# Sobel Edge Detection

sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

# Prewitt Edge Detection

kernelx = np.array([[1,0,-1],
                    [1,0,-1],
                    [1,0,-1]])

kernely = np.array([[1,1,1],
                    [0,0,0],
                    [-1,-1,-1]])

prewittx = cv2.filter2D(image, -1, kernelx)
prewitty = cv2.filter2D(image, -1, kernely)
prewitt = prewittx + prewitty

# Laplacian Edge Detection

laplacian = cv2.Laplacian(image, cv2.CV_64F)

# Canny Edge Detection

canny = cv2.Canny(image, 100, 200)

titles = ['Original Image', 'Sobel',
          'Prewitt', 'Laplacian', 'Canny']

images = [image, sobel, prewitt, laplacian, canny]

plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()

cv2.imwrite('/workspaces/Edge-detection-system/sobel_output.jpg', sobel)
cv2.imwrite('/workspaces/Edge-detection-system/prewitt_output.jpg', prewitt)
cv2.imwrite('/workspaces/Edge-detection-system/laplacian_output.jpg', laplacian)
cv2.imwrite('/workspaces/Edge-detection-system/canny_output.jpg', canny)
