import torch
import numpy as np
#vgg has 13 conv_layers,only prune conv_layer
layer_begin=0
layer_end=12
layer_inter=1
last_index=13
class Mask(torch.nn.Module):
    def __init__(self,model):
        super(Mask,self).__init__()
        self.model_size={}
        self.model_length={}
        self.sparsity={}
        self.mat={}
        self.model=model
        self.mask_index=[]
    def get_codebook_filter(self,weight_torch,sparsity,length):
        codebook=np.ones(length)
        if len(weight_torch.size())==4:
            compress_num=int(weight_torch.size()[0]*sparsity)
            weight_torch1=weight_torch.view(weight_torch.size()[0],-1)
            norm=torch.norm(weight_torch1,1,1)
            norm_np=norm.cpu().numpy()
            pruning_index=np.argsort(norm_np)[:compress_num]
            kernel_length=weight_torch.size()[1]*weight_torch.size()[2]*weight_torch.size()[3]
            for i in range(len(pruning_index)):
                codebook[pruning_index[i]*kernel_length:(pruning_index[i]+1)*kernel_length]=0
            print("filter codebook done!")
        else:
            pass
        return codebook
        
    def convert_to_tensor(self,x):
        tensor=torch.FloatTensor(x)
        return tensor.cuda()

    def init_length(self):
        for index,item in enumerate(self.model.parameters()):
            
            self.model_size[index]=item.size()
        
        for index1 in self.model_size:
            for index2 in range(len(self.model_size[index1])):
                if index2==0:
                    self.model_length[index1]=self.model_size[index1][0]
                else:
                    self.model_length[index1]*=self.model_size[index1][index2]
    
    def init_rate(self,layer_rate):
        for index,item in enumerate(self.model.parameters()):
            self.sparsity[index]=1
        for key in range(layer_begin,layer_end+1,layer_inter):
            self.sparsity[key]=layer_rate
        self.mask_index=[x for x in range(0,last_index,1)]

    def init_mask(self,layer_rate):
        self.init_rate(layer_rate)
        for index,item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # self.mat[index]=self.get_codebook(item.data,self.sparsity[index],self.model_length[index])
                self.mat[index]=self.get_codebook_filter(item.data,self.sparsity[index],self.model_length[index])
                self.mat[index]=self.convert_to_tensor(self.mat[index])
                self.mat[index]=self.mat[index].cuda()
        print("mask ready")
    
    def do_mask(self):
        for index,item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a=item.data.view(self.model_length[index])
                b=a*self.mat[index]
                item.data=b.view(self.model_size[index])
        print("mask done")
    
    def if_zero(self):
        for index,item in enumerate(self.model.parameters()):
            if index in [x for x in range(layer_begin,layer_end+1,layer_inter)]:
                a=item.data.view(self.model_length[index])
                b=a.cpu().numpy()
                print("layer: %d, number of nonzero weight is %d, zero is %d" % (
                    index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))