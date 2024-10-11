
import cv2
from utils.bbox import *
frame = cv2.imread('/home/wangzimo/VTT/VTT/tmp.jpg')

print(hand_pic2bbox(frame))