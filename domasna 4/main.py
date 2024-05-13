import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def getImgPaths(directory):
    imgPaths = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            imgPaths.append(os.path.join(directory, filename))

    return imgPaths


def getImgs(imgPaths):
    imgs = []
    for directory in imgPaths:
        img = cv2.imread(directory)
        imgs.append((directory, img))

    return imgs

def bitWiseOR(list):
    result = list[0]
    for i in range(1, len(list)):
        result = cv2.bitwise_or(result, list[i])
    return result

def segmentateImage(image):
    imgBlurred = cv2.medianBlur(image, 3)
    imgGray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2HSV)
    seg1 = 255 - cv2.threshold(imgGray, 64, 255, cv2.THRESH_BINARY)[1]
    seg2 = cv2.inRange(imgHSV, (0, 48, 0), (180, 224, 255))

    segmented = bitWiseOR([seg1, seg2])

    contours = cv2.findContours(segmented, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    width, height = segmented.shape
    result = np.zeros((width, height, 3), np.uint8)
    cv2.drawContours(result, contours, 0, (255, 255, 255), -1)

    return (result, contours)


class Sample:
    def __init__(self, path, image, segmentedImg, contours):
        self.path = path
        self.image = image
        self.segmentedImage = segmentedImg
        self.contours = contours


def main():
    queryImgPaths = getImgPaths("query")
    databaseImgPaths = getImgPaths("database")
    queryImgs = getImgs(queryImgPaths)
    databaseImgs = getImgs(databaseImgPaths)
    querySamples = []
    databaseSamples = []

    for image in queryImgs:
        segmented = segmentateImage(image[1])
        querySamples.append(Sample(image[0], image[1], segmented[0], segmented[1]))

    for image in databaseImgs:
        segmented = segmentateImage(image[1])
        databaseSamples.append(Sample(image[0], image[1], segmented[0], segmented[1]))

    results = []
    for i, querySample in enumerate(querySamples):
        perSampleResults = []
        for j, databaseSample in enumerate(databaseSamples):
            cnt1 = querySample.contours[0]
            cnt2 = databaseSample.contours[0]

            res1 = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0) + 1
            res2 = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I2, 0.0) + 1
            res3 = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I3, 0.0) + 1

            result = res1 * res1 + res2 * res2 + res3 * res3
            perSampleResults.append((i, j, result))
        perSampleResults.sort(key=lambda x: x[2], reverse=False)
        results.append(perSampleResults)

    for result in results:
        figure = plt.figure(querySamples[result[0][0]].path)

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(querySamples[result[0][0]].image, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("1. Query Image")

        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(databaseSamples[result[0][1]].image, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("1st best macth")

        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(databaseSamples[result[1][1]].image, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("2nd best match")

        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(databaseSamples[result[2][1]].image, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("3rd best match")

        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(databaseSamples[result[3][1]].image, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("4th best match")

        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(databaseSamples[result[len(results)][1]].image, cv2.COLOR_BGR2RGB))
        plt.xticks([]), plt.yticks([])
        plt.title("Worst match")

        plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()