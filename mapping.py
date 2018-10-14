import re
import matplotlib.pyplot as plt
import numpy as np
import os
#利用正则表达式提取数据进行制图
def get_filename(root='./',pattern='slurm'):
    for root,dirs,files in os.walk(root):
        for filename in files:
            if pattern in filename:
                return filename

def get_dirname(root='../',pattern='tempfile'):
    for root,dirs,files in os.walk(root):
        for dirname in dirs:
            if pattern in dirname:
                return dirname

def get_num(string):
    pattern=re.compile('[0-9].*$')
    a=pattern.findall(string)
    return float(a[0])

def get_start_epoch(filename):
    pattern=re.compile('---------Epoch.*')
    file=open(filename,'r').read()
    a=pattern.findall(file)
    a=re.search('[0-9].*?',a[0]).group()
    print(int(a))
    return int(a)

def get_data(filename):
    file=open(filename,'r').read()
    pattern1=re.compile('train accuracy rate.*')
    pattern2=re.compile('validation accuracy rate.*')
    # pattern3=re.compile('LR.*')
    train_acc=pattern1.findall(file)
    validation_acc=pattern2.findall(file)
    # learning_rate=pattern3.findall(file)
    for i in range(len(train_acc)):
        if len(train_acc)==40:
            train_acc[i]=float(train_acc[i][32:38])
        else:
            train_acc[i]=float(train_acc[i][32:38])
    for i in range(len(validation_acc)):
        if len(validation_acc)==40:
            validation_acc[i]=float(validation_acc[i][32:38])
        else:
            validation_acc[i]=float(validation_acc[i][32:38])
    # for i in range(len(learning_rate)):
    #     learning_rate[i]=get_num(learning_rate[i])

    over=np.array(train_acc)-np.array(validation_acc)
    # return train_acc,validation_acc,over,learning_rate
    return train_acc,validation_acc,over
def plot_slurm(filename,filename2=None):
    if filename2==None:
        train_acc,validation_acc,over=get_data(filename)
        epoch=np.arange(1,len(train_acc)+1)
        plt.plot(epoch,train_acc,label='train_acc')
        plt.plot(epoch,validation_acc,label='validation_acc')
        plt.xlabel('epoch')
        plt.ylabel('acc_rate')
        plt.title('train_acc and validation_acc')
        plt.legend()
        plt.show()

        plt.plot(epoch,over)
        plt.xlabel('epoch')
        plt.ylabel('difference')
        plt.title('the difference of train_acc and validation_acc')
        plt.show()

        # plt.plot(epoch,learning_rate)
        # plt.xlabel('epoch')
        # plt.ylabel('learning_rate')
        # plt.show()

