import cv2
import numpy as np

def contrast_stretching(img, points):

    new_img = np.zeros(img.shape, img.dtype)


    for channel in range(img.shape[2]):

        img_channel = img[:,:,channel]


        lut = np.zeros(256, dtype=np.uint8)


        points.sort(key=lambda x: x[0])


        for i in range(1, len(points)):
            x1, y1 = points[i-1]
            x2, y2 = points[i]
            for x in range(x1, x2+1):
                lut[x] = int((x - x1) * (y2 - y1) / (x2 - x1) + y1)


        new_img[:,:,channel] = cv2.LUT(img_channel, lut)

    return new_img


img = cv2.imread('img1.png')


points = [(0, 0), (70, 20), (140, 200), (255, 255)]


new_img = contrast_stretching(img, points)
cv2.imshow("New image",new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imwrite('new_image.jpg', new_img)
