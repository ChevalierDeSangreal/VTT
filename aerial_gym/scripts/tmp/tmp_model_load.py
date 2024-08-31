import sys
sys.path.append('/home/zim/Documents/python/AGAPG-main')
from aerial_gym.models import TrackGroundModelVer5

model = TrackGroundModelVer5()
for i in model.state_dict():
    print(i)