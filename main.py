import cv2, pytesseract, glob
import numpy as np

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def preprocess(image):
    image = get_grayscale(image)
    image = cv2.GaussianBlur(image, (5,5), 0)
    image = cv2.dilate(image, (5,5))
    image = cv2.erode(image, (5,5))
    contours = cv2.findContours(image,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, [cv2.convexHull(contours)], -1, (0, 255, 0), 3)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    return image

if __name__ == "__main__":
    imgs = []
    for f in glob.glob('./figs/*.jpg'):
        imgs.append(preprocess(cv2.imread(f)))


    custom_config = r'--oem 3 --psm 6'

    for img in imgs:
        h, w = img.shape
        boxes = pytesseract.image_to_boxes(img)
        for b in boxes.splitlines():
            b = b.split(' ')
            img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

        cv2.imshow('img', img)
        cv2.waitKey(0)

        print(pytesseract.image_to_string(img, config=custom_config))
        print("sss")