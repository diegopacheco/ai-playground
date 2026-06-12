import cv2
import numpy as np

SIZE = 400


def _canvas():
    return np.zeros((SIZE, SIZE), np.uint8)


def rabbit():
    img = _canvas()
    cv2.ellipse(img, (200, 270), (85, 75), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (160, 130), (24, 100), -12, 0, 360, 255, -1)
    cv2.ellipse(img, (238, 125), (24, 105), 14, 0, 360, 255, -1)
    return img


def dog():
    img = _canvas()
    cv2.ellipse(img, (210, 230), (95, 70), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (105, 250), (55, 28), 8, 0, 360, 255, -1)
    cv2.ellipse(img, (265, 160), (28, 60), 25, 0, 360, 255, -1)
    cv2.ellipse(img, (95, 285), (18, 10), 0, 0, 360, 255, -1)
    return img


def bird():
    img = _canvas()
    cv2.ellipse(img, (200, 250), (90, 55), 0, 0, 360, 255, -1)
    pts = np.array([[150, 230], [60, 110], [185, 200]], np.int32)
    cv2.fillPoly(img, [pts], 255)
    pts = np.array([[250, 230], [340, 110], [215, 200]], np.int32)
    cv2.fillPoly(img, [pts], 255)
    pts = np.array([[285, 250], [345, 240], [285, 270]], np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img


def swan():
    img = _canvas()
    cv2.ellipse(img, (230, 290), (100, 60), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (240, 200), (80, 80), 0, 90, 235, 255, 30)
    cv2.circle(img, (194, 134), 26, 255, -1)
    pts = np.array([[205, 122], [265, 134], [205, 148]], np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img


def snail():
    img = _canvas()
    cv2.ellipse(img, (200, 300), (130, 35), 0, 0, 360, 255, -1)
    cv2.circle(img, (220, 230), 75, 255, -1)
    cv2.ellipse(img, (105, 225), (14, 75), -18, 0, 360, 255, -1)
    cv2.ellipse(img, (138, 228), (14, 70), 8, 0, 360, 255, -1)
    return img


PUPPETS = {
    "rabbit": rabbit,
    "dog": dog,
    "bird": bird,
    "swan": swan,
    "snail": snail,
}


def silhouette(name):
    return PUPPETS[name]()


def main_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def solidity(contour):
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0:
        return 0.0
    return cv2.contourArea(contour) / hull_area


def build_references():
    refs = {}
    for name, draw in PUPPETS.items():
        contour = main_contour(draw())
        refs[name] = (contour, solidity(contour))
    return refs
