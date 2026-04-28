import cv2
import urllib.request
import numpy as np
import matplotlib as plt

#leer imagen
img=cv2.imread("1019.jpg",0)

#otsu
ret,thresh=cv2.threshold(img,0,255,cv2.THRESH_BINARY +cv2.THRESH_OTSU)

cv2.imshow("Otsu", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#k-means
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
pixel_vals=img.reshape((-1,3))
pixel_vals=np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape((img.shape))
cv2.imshow("K-means",segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#contornos activos
#transformando
image = cv2.imread('1019.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

s = np.linspace(0, 2*np.pi, 400)
r = 100 + 100 * np.sin(s) # Centro Y
c = 220 + 100 * np.cos(s) # Centro X
# Tip: Cambia el 100 por el radio que quieras y los valores 100/220 por la posición
init = np.array([r, c]).T

# alpha: elasticidad (más alto = más rígida)
# beta: suavidad (más alto = menos esquinas)
# gamma: velocidad de ajuste
#snake = active_contour(blurred, init, alpha=0.015, beta=10, gamma=0.001)

_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    #encontrar contornos
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #dibujar contornos
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

cv2.imshow('Contornos', image)
cv2.waitKey(0)
cv2.destroyAllWindows()