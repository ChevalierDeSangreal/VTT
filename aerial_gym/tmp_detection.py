import sys

sys.path.append('/home/wangzimo/VTT/VTT')
# from aerial_gym.utils import *

from utils.bbox import *
from pysott.tools import *

frame = cv2.imread('/home/wangzimo/VTT/VTT/tmp.jpg')
bbox = hand_pic2bbox(frame)

detnet = Detnet(frame, bbox)
new_bbox = detnet(frame)
print(bbox, new_bbox)

new_bbox = list(map(int, new_bbox))
cv2.rectangle(frame, (new_bbox[0], new_bbox[1]), (new_bbox[0]+new_bbox[2], new_bbox[1]+new_bbox[3]), (0, 255, 0), 3)
cv2.imshow('tmp', frame)
cv2.waitKey(4000)
