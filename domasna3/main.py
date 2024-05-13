import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def getImgsPaths(directories):
    imgPaths = []

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                imgPaths.append(os.path.join(directory, filename))
    return imgPaths


def getImgs(imgPaths):
    imgs = []
    for directory in imgPaths:
        image = cv2.imread(directory)
        imgs.append((directory, image))
    return imgs

def bitwOR(list):
    result = list[0]
    for i in range(1, len(list)):
        result = cv2.bitwise_or(result, list[i])
    return result

def segmentImage(image):
    imgBlurred = cv2.medianBlur(image,3)
    imgGray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2HSV)
    segment1 = 255 - cv2.threshold(imgGray, 64, 255, cv2.THRESH_BINARY)[ 1]
    segment2 = cv2.inRange(imgHSV, (0, 48, 0), (180, 224, 255))

    segmentedImg = bitwOR([segment1, segment2])

    contours = cv2.findContours(segmentedImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea,reverse=True)

    width, height = segmentedImg.shape
    result = np.zeros((width, height, 3), np.uint8)
    cv2.drawContours(result, contours, 0, (255, 255, 255), -1)

    return (result, contours)


def main():
    imgPaths = getImgsPaths(("database", "database1"))
    imgs = getImgs(imgPaths)

    for image in imgs:
        segmented, contours = segmentImage(image[1])
        imgContours = image[1].copy()
        cv2.drawContours(imgContours, contours, 0, (0, 0, 255), 2)

        figure = plt.figure(image[0])

        plt.subplot(3, 1, 1)
        plt.imshow(cv2.cvtColor(image[1], cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("Оригинална слика")

        plt.subplot(3, 1, 2)
        plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("Црно-бела слика добиена со сегментација")

        plt.subplot(3, 1, 3)
        plt.imshow(cv2.cvtColor(imgContours, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("Контура која го дефинира листот")

        plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
