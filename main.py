import cv2, pytesseract, glob
import numpy as np

def deskew(img):
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def thresh_callback(val, img):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(img, threshold, threshold * 2)
    # Find contours
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i], False)
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (250,50,50)
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)
    # Show in a window
    cv2.imshow('Contours', drawing)

def preprocess(src):
    #img_ = cv2.resize(src, (1024,720))
    img_ = get_grayscale(src)
    img = cv2.GaussianBlur(img_, (5,5), 0)
    img = cv2.dilate(img, (5,5))
    img = cv2.erode(img, (5,5))
    edges = cv2.Canny(img, 180, 255)
    # Find contours
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [cnt], 0, 255, -1)

    (y, x) = np.where(mask == 255)

    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    img = img[topy:bottomy + 1, topx:bottomx + 1]

    """cv2.imshow('img', img)
    cv2.waitKey(0)"""
    return img_

if __name__ == "__main__":
    imgs = []
    for f in glob.glob('./figs/*'):
        imgs.append(preprocess(cv2.imread(f)))

    custom_config = r'--oem 3 --psm 6'

    for img in imgs:
        print(img.shape)
        h, w = img.shape
        print(pytesseract.image_to_string(img, config=custom_config))