import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time
import numpy as np
import math
learning_rate=0.01
train_epochs=500
weight_decay=0.0004
lr_decay=0
milestones=[50,200,300,400]
print("----------------------hyperparameter---------------------------")
print("learning rate {:f}".format(learning_rate))
print("weight decay {:f}".format(weight_decay))
print("learning dacay {:f}".format(lr_decay))
print("---------------------------------------------------------------")
class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}
def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                        shuffle=True, num_workers=4)  
validationset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=32,
                                        shuffle=False, num_workers=4)
net=vgg16()
net._initialize_weights()
# for layer in net.parameters():
#     print(layer.data)

# pretrained_dict=torch.load('vgg16-397923af.pth')
# net_dict=net.state_dict()
# pretrained_dict={k:v for k,v in pretrained_dict.items() if k in net_dict}
# net_dict.update(pretrained_dict)
# net.load_state_dict(net_dict)
net.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=learning_rate,weight_decay=weight_decay)
#scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones,gamma=lr_decay)
def train_epoch():
    # start_time=time.time()
    training_loss=0.0
    total_image=0
    train_correct=0
    for index,data in enumerate(trainloader,0):
        inputs,labels=data
        #copy my tensor to gpus
        inputs,labels=inputs.cuda(),labels.cuda()
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        #scheduler.step()
        optimizer.step()
        _,predicted=torch.max(outputs,1)
        total_image+=labels.size(0)
        train_correct+=(predicted==labels).sum().item()
        training_loss+=loss.item()
    return training_loss/(index+1),int(100*train_correct/total_image)
def validation_epoch():
    validation_loss=0.0
    validation_correct=0
    total_image=0
    with torch.no_grad():
        for index,data in enumerate(validationloader,0):
            inputs,labels=data
            inputs,labels=inputs.cuda(),labels.cuda()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            _,predicted=torch.max(outputs,1)
            validation_loss+=loss.item()
            total_image+=labels.size(0)
            validation_correct+=(predicted==labels).sum().item()
    return validation_loss/(index+1),int(100*validation_correct/total_image)
def main():
    for epoch in range(train_epochs):
        start_time=time.time()
        tra_loss,tra_acc=train_epoch()
        val_loss,val_acc=validation_epoch()
        end_time=time.time()-start_time
        print("Epoch "+str(epoch + 1)+" of "+str(train_epochs)+" took   "+str(end_time)+"s")
        print("  training loss:                   "+str(tra_loss))
        print("  train accuracy rate:            "+str(tra_acc)+"%")
        print("  validation loss:                 "+str(val_loss))
        print("  validation accuracy rate:       "+str(val_acc)+"%")
    torch.save(net.state_dict(),'vgg16_parametes.pth')
    
if __name__ == '__main__':
    main()