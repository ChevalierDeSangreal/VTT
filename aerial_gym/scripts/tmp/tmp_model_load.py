import sys
sys.path.append('/home/wangzimo/VTT/VTT')
from aerial_gym.models import TrackGroundModelVer5

model = TrackGroundModelVer5()
for i in model.state_dict():
    print(i)