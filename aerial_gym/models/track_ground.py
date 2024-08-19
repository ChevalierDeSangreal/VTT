import torch
import torch.nn as nn
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchsummary import summary
from .resnet import Resnet

class TrackGroundModel(nn.Module):
    """
    
    """
    def __init__(self, input_size=16, hidden_size1=128, hidden_size2=128, hidden_size3=128, hidden_size4=128, output_size=4, device='cpu'):
        print("TrackGroundModel Initializing...")

        super(TrackGroundModel, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.hidden_layer4 = nn.Linear(hidden_size3, hidden_size4).to(device)
        self.activation4 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size4, output_size).to(device)

        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 4)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, image):
        
        # (batch_size, H, W, 4) to (batch_size, 3, H, W)
        image = image[:, :, :, :3].permute(0, 3, 1, 2).float()
        # print(image.shape, type(image[0, 0, 0, 0]))

        feature = self.resnet(image)
        x = torch.cat((now_state, feature), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.hidden_layer4(x)
        x = self.activation4(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x

class TrackGroundModelVer2(nn.Module):
    """
    
    """
    def __init__(self, input_size=16, hidden_size1=64, hidden_size2=64, hidden_size3=64, output_size=4, device='cpu'):
        print("TrackGroundModel Initializing...")

        super(TrackGroundModelVer2, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size3, output_size).to(device)

        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 4)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, image):
        
        # (batch_size, H, W, 4) to (batch_size, 3, H, W)
        image = image[:, :, :, :3].permute(0, 3, 1, 2).float()
        # print(image.shape, type(image[0, 0, 0, 0]))

        feature = self.resnet(image)
        x = torch.cat((now_state, feature), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x
    
class TrackGroundModelVer3(nn.Module):
    """
    
    """
    def __init__(self, input_size=13, hidden_size1=64, hidden_size2=64, hidden_size3=64, output_size=4, device='cpu'):
        print("TrackGroundModel Initializing...")

        super(TrackGroundModelVer3, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size3, output_size).to(device)

        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 4)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, image):
        
        # (batch_size, H, W, 4) to (batch_size, 3, H, W)
        image = image[:, :, :, :3].permute(0, 3, 1, 2).float()
        # print(image.shape, type(image[0, 0, 0, 0]))

        feature = self.resnet(image)
        x = torch.cat((now_state, feature), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x
    
class TrackGroundModelVer4(nn.Module):
    """
    
    """
    def __init__(self, input_size=13, hidden_size1=256, hidden_size2=256, hidden_size3=256, hidden_size4=256, hidden_size5=256, output_size=4, device='cpu'):
        print("TrackGroundModel Initializing...")

        super(TrackGroundModelVer4, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.hidden_layer4 = nn.Linear(hidden_size3, hidden_size4).to(device)
        self.activation4 = nn.ReLU().to(device)
        self.hidden_layer5 = nn.Linear(hidden_size4, hidden_size5).to(device)
        self.activation5 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size5, output_size).to(device)
        
        torch.nn.init.kaiming_normal_(self.hidden_layer1.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer2.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer3.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer4.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer5.weight)
        torch.nn.init.kaiming_normal_(self.output_layer.weight)
        
        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 4)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, image):
        
        # (batch_size, H, W, 4) to (batch_size, 3, H, W)
        image = image[:, :, :, :3].permute(0, 3, 1, 2).float()
        # print(image.shape, type(image[0, 0, 0, 0]))

        feature = self.resnet(image)
        x = torch.cat((now_state, feature), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.hidden_layer4(x)
        x = self.activation4(x)
        x = self.hidden_layer5(x)
        x = self.activation5(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x
    
class TrackGroundModelVer5(nn.Module):
    """
    
    """
    def __init__(self, input_size=13, hidden_size1=128, hidden_size2=128, hidden_size3=128, 
                 hidden_size4=128, output_size=4, device='cpu', resnet_load_path='/home/cgv841/wzm/FYP/AGAPG/aerial_gym/param_saved/resnet.pth'):
        print("TrackGroundModel Initializing...")

        super(TrackGroundModelVer5, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.hidden_layer4 = nn.Linear(hidden_size3, hidden_size4).to(device)
        self.activation4 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size4, output_size).to(device)

        self.resnet = Resnet()
        if resnet_load_path:
            print("Loading resnet from:", resnet_load_path)
            self.resnet.load_state_dict(torch.load(resnet_load_path))
        self.resnet.dense3 = nn.Linear(16, 4).to(device)
        # self.transform = transforms.Compose([
        #     transforms.Resize(227),
        #     transforms.ToTensor(),  
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


    def forward(self, now_state, image):
        
        # (batch_size, H, W, 4) to (batch_size, 3, H, W)
        image = image[:, :, :, :3].permute(0, 3, 1, 2).float()
        # image = self.transform(image)
        feature = self.resnet(image)
        
        x = torch.cat((now_state, feature), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.hidden_layer4(x)
        x = self.activation4(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x
    
class TrackGroundModelVer6(nn.Module):
    """
    
    """
    def __init__(self, input_size=12, hidden_size1=256, hidden_size2=256, hidden_size3=256, hidden_size4=256, hidden_size5=256, output_size=4, device='cpu'):
        print("TrackGroundModel Initializing...")

        super(TrackGroundModelVer6, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ReLU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ReLU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ReLU().to(device)
        self.hidden_layer4 = nn.Linear(hidden_size3, hidden_size4).to(device)
        self.activation4 = nn.ReLU().to(device)
        self.hidden_layer5 = nn.Linear(hidden_size4, hidden_size5).to(device)
        self.activation5 = nn.ReLU().to(device)
        self.output_layer = nn.Linear(hidden_size5, output_size).to(device)

        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, tar_pos):
        

        x = torch.cat((now_state, tar_pos), dim=1)
        x = self.hidden_layer1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        x = self.activation3(x)
        x = self.hidden_layer4(x)
        x = self.activation4(x)
        x = self.hidden_layer5(x)
        x = self.activation5(x)
        x = self.output_layer(x)
        # x = self.tanh(x)
        x = torch.sigmoid(x) * 2 - 1
        return x

class TrackGroundModelVer7(nn.Module):
    """
    Added Bn and replace ReLU with ELU
    """
    def __init__(self, input_size=13, hidden_size1=256, hidden_size2=256, hidden_size3=256, hidden_size4=256, hidden_size5=256, output_size=4, device='cpu'):
        print("TrackGroundModel Initializing...")

        super(TrackGroundModelVer7, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1).to(device)
        self.activation1 = nn.ELU().to(device)
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.activation2 = nn.ELU().to(device)
        self.hidden_layer3 = nn.Linear(hidden_size2, hidden_size3).to(device)
        self.activation3 = nn.ELU().to(device)
        self.hidden_layer4 = nn.Linear(hidden_size3, hidden_size4).to(device)
        self.activation4 = nn.ELU().to(device)
        self.hidden_layer5 = nn.Linear(hidden_size4, hidden_size5).to(device)
        self.batch_norm5 = nn.BatchNorm1d(hidden_size5).to(device)
        self.activation5 = nn.ELU().to(device)
        self.output_layer = nn.Linear(hidden_size5, output_size).to(device)

        torch.nn.init.kaiming_normal_(self.hidden_layer1.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer2.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer3.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer4.weight)
        torch.nn.init.kaiming_normal_(self.hidden_layer5.weight)
        torch.nn.init.kaiming_normal_(self.output_layer.weight)

        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 4)
        # self.tanh = nn.Tanh().to(device)

    def forward(self, now_state, image):
        # (batch_size, H, W, 4) to (batch_size, 3, H, W)
        image = image[:, :, :, :3].permute(0, 3, 1, 2).float()

        feature = self.resnet(image)
        x = torch.cat((now_state, feature), dim=1)
        x = self.hidden_layer1(x)
        # x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.hidden_layer2(x)
        # x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.hidden_layer3(x)
        # x = self.batch_norm3(x)
        x = self.activation3(x)
        x = self.hidden_layer4(x)
        # x = self.batch_norm4(x)
        x = self.activation4(x)
        x = self.hidden_layer5(x)
        x = self.batch_norm5(x)
        x = self.activation5(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x) * 2 - 1
        return x

if __name__ == "__main__":
    # 获取 ResNet-18 模型
    # resnet18_model = models.resnet18(pretrained=True)
    # in_features = resnet18_model.fc.in_features
    # resnet18_model.fc = nn.Linear(in_features, 4)
    # summary(resnet18_model.to('cuda:0'), input_size=(3, 227, 227))

    # # 输出模型结构
    # print(resnet18_model)
    # alexnet = models.alexnet(pretrained=True).to('cpu')
    # in_features = alexnet.classifier[6].in_features
    # alexnet.classifier[6] = nn.Linear(in_features, 4)
    # summary(alexnet.to('cuda:0'), input_size=(3, 227, 227))
    # print(alexnet)
    
    resnet = Resnet()
    summary(resnet.to("cuda:0"), input_size=(3, 227, 227))