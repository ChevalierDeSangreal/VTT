
import cv2
from utils.bbox import *
frame = cv2.imread('/home/zim/Documents/python/AGAPG-main/tmp.jpg')

print(hand_pic2bbox(frame))