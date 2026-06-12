import sys
import cv2
import numpy as np
from shapes import build_references, main_contour, solidity

MATCH_LIMIT = 0.45
MIN_AREA_RATIO = 0.02


def segment(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def classify(contour, refs):
    s = solidity(contour)
    best_name, best_score = None, float("inf")
    for name, (ref, ref_solidity) in refs.items():
        score = cv2.matchShapes(contour, ref, cv2.CONTOURS_MATCH_I1, 0.0)
        score += 2.0 * abs(s - ref_solidity)
        if score < best_score:
            best_name, best_score = name, score
    return best_name, best_score


def annotate(frame, refs):
    mask = segment(frame)
    contour = main_contour(mask)
    if contour is None:
        return frame
    area = cv2.contourArea(contour)
    if area < MIN_AREA_RATIO * frame.shape[0] * frame.shape[1]:
        return frame
    name, score = classify(contour, refs)
    matched = score < MATCH_LIMIT
    color = (80, 200, 80) if matched else (60, 60, 220)
    label = f"{name} ({score:.3f})" if matched else "no puppet"
    cv2.drawContours(frame, [contour], -1, color, 3)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, label, (x, max(30, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    return frame


def run_image(src, dst):
    refs = build_references()
    frame = cv2.imread(src)
    if frame is None:
        print(f"could not read {src}")
        sys.exit(1)
    out = annotate(frame, refs)
    cv2.imwrite(dst, out)
    print(f"wrote {dst}")


def run_camera():
    refs = build_references()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open camera")
        sys.exit(1)
    print("press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        out = annotate(frame, refs)
        cv2.imshow("puppet shadows", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--image":
        run_image(sys.argv[2], sys.argv[3])
    else:
        run_camera()
