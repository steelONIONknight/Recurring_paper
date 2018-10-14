import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import time
import numpy as np
import math
from pruning import Mask
from dense_training import vgg16

learning_rate=0.01
weight_decay=0.0004
learning_decay=0
train_epochs=500
epoch_prune=1
sparsity=0.4
print("----------------------hyperparameter---------------------------")
print("learning rate {:f}".format(learning_rate))
print("weight decay {:f}".format(weight_decay))
print("learning dacay {:f}".format(learning_decay))
print("sparsity {:f}".format(sparsity))
print("---------------------------------------------------------------")
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
mynet=vgg16()

checkpoint=torch.load('vgg16_parameters.pth')
# print("---------------------output pretrained data-------------------------")
# print(checkpoint)
mynet.load_state_dict(checkpoint)
mynet.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(mynet.parameters(),lr=learning_rate,weight_decay=weight_decay)
m=Mask(mynet)
def train_epoch():
    training_loss=0.0
    total_image=0
    train_correct=0
    mynet.train()
    for index,data in enumerate(trainloader,0):
        inputs,labels=data
        #copy my tensor to gpus
        inputs,labels=inputs.cuda(),labels.cuda()
        optimizer.zero_grad()
        outputs=mynet(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        #scheduler.step()
        optimizer.step()
        _,predicted=torch.max(outputs,1)
        total_image+=labels.size(0)
        train_correct+=(predicted==labels).sum().item()
        training_loss+=loss.item()
    return training_loss/(index+1),100.0*train_correct/total_image
def validation_epoch():
    validation_loss=0.0
    validation_correct=0
    total_image=0
    mynet.eval()
    with torch.no_grad():
        for index,data in enumerate(validationloader,0):
            inputs,labels=data
            inputs,labels=inputs.cuda(),labels.cuda()
            outputs=mynet(inputs)
            loss=criterion(outputs,labels)
            _,predicted=torch.max(outputs,1)
            validation_loss+=loss.item()
            total_image+=labels.size(0)
            validation_correct+=(predicted==labels).sum().item()
    return validation_loss/(index+1),100.0*validation_correct/total_image
def main():
    global mynet
    for epoch in range(train_epochs):
        start_time=time.time()
        tra_loss,tra_acc=train_epoch()
        val_loss,val_acc=validation_epoch()
        if (epoch%epoch_prune==0 or epoch==train_epochs-1):
            m.model=mynet
            m.init_length()
            m.init_mask(sparsity)
            m.do_mask()
            m.if_zero()
            mynet=m.model
            mynet=mynet.cuda()
        end_time=time.time()-start_time
        print("  ---------Epoch {:d} of {:d} took {:.3f}s-----".format(epoch+1,train_epochs,end_time))
        print("  training loss:                   {:.4f}".format(tra_loss))
        print("  train accuracy rate:            {:.3f}%".format(tra_acc))
        print("  validation loss:                 {:.4f}".format(val_loss))
        print("  validation accuracy rate:       {:.3f}%".format(val_acc))
    torch.save(mynet.state_dict(),'vgg16_sparse_training_parameters.pth')
if __name__=='__main__':
    main()
