import numpy as np
import math
import cv2

def find_distance(w, h, posr, posl):
    baseline = 9
    f_pixel = 6
    alpha = 56.6

    f_pixel = (w * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

    xr = posr[0]
    xl = posl[0]

    disp = xl - xr
    depth = (baseline * f_pixel) / disp

    distance = abs(depth) * 0.303881 -0.43233
    # Sanity check
    if math.isinf(distance):
        distance = 0

    return distance


def find_angle(w, h, posr, posl, fov=90):
    xr, yr = posr
    xl, yl = posl

    return fov / w * ((xr + xl) / 2 - h)


def getTranslationDeltas(imgW, imgH, xl1, yl1, xr1, yr1, xl2, yl2, xr2, yr2):
    d = find_distance(imgW // 2, imgH, (xr1, yr1), (xl1, yl1))
    alpha = -find_angle(imgW, imgH, (xr1, yr1), (xl1, yl1))
    dprime = find_distance(imgW // 2, imgH, (xr2, yr2), (xl2, yl2))
    beta = -find_angle(imgW, imgH, (xr2, yr2), (xl2, yl2))
    alpha = (alpha / 180) * np.pi # Radians
    beta = (beta / 180) * np.pi # Radians
    alpha += 3*np.pi/2  # For sin(), cos()
    beta += 3*np.pi/2  # For sin(), cos()
    dx = dprime*np.cos(beta) - d*np.cos(alpha)
    dy = dprime*np.sin(beta) - d*np.sin(alpha)
    return dx, dy
