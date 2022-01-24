import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)



def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ft_net(nn.Module):

    def __init__(self, class_num, stride=1, circle=False):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=False)
        model_ft.load_state_dict(torch.load('D:/NEW_HDGAN/FINAL_TEST/CODE/PIG-MM/train/train_gan/resnet50.pth'))   #you can also use pretrained= True and annotate this line of code
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_ftrs = model_ft.fc.in_features
        self.model = model_ft
        self.circle = circle

        self.bottleneck = nn.BatchNorm1d(self.num_ftrs)
        self.bottleneck.bias.requires_grad_(False)
        self.fc = nn.Linear(self.num_ftrs, class_num, bias=True)
        self.bottleneck.apply(weights_init_kaiming)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        # x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)  # -> 512 32*16
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # f = self.model.partpool(x)  # 8 * 2048 4*1
        x = self.model.avgpool(x)  # 8 * 2048 1*1
        x = x.view(x.size(0), x.size(1))
        f = x
        out = self.bottleneck(x)
        out = self.fc(out)
        return out, f
