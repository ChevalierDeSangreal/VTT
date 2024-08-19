import sys
sys.path.append('/home/cgv841/wzm/FYP/AGAPG')
from aerial_gym.models import TrackGroundModelVer5

model = TrackGroundModelVer5()
for i in model.state_dict():
    print(i)