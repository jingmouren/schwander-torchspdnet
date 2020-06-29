import sys
sys.path.insert(0,'..')
from pathlib import Path
import os
import random

import numpy as np
import torch as th
import torch.nn as nn
from torch.utils import data

import torchspdnet.nn as spdnet
from torchspdnet.optimizers import MixOptimizer

# Definition of the SPDNet
class AfewNet(nn.Module):
    def __init__(self, bn=False):
        super(__class__,self).__init__()
        self._bn = bn
        dim=400
        dim1=200; dim2=100; dim3=50
        classes=7
        self.re=spdnet.ReEig()
        self.bimap1=spdnet.BiMap(1,1,dim,dim1)
        if bn:
            self.batchnorm1=spdnet.BatchNormSPD(dim1)
        self.bimap2=spdnet.BiMap(1,1,dim1,dim2)
        if bn:
            self.batchnorm2=spdnet.BatchNormSPD(dim2)
        self.bimap3=spdnet.BiMap(1,1,dim2,dim3)
        if bn:
            self.batchnorm3=spdnet.BatchNormSPD(dim3)
        self.logeig=spdnet.LogEig()
        self.linear=nn.Linear(dim3**2,classes).double()
    def forward(self,x):
        x=self.bimap1(x)
        if self._bn:
            x=self.batchnorm1(x)
        x=self.re(x)
        x=self.bimap2(x)
        if self._bn:
            x=self.batchnorm2(x)
        x=self.re(x)
        x=self.bimap3(x)
        if self._bn:
            x=self.batchnorm1(x)
        x=self.logeig(x)
        x_vec=.view(x.shape[0],-1)
        y=self.linear(x_vec)
        return y

def afew(data_loader):
    #main parameters
    lr=1e-2
    data_path='data/afew/'
    n=400 #dimension of the data
    C=7 #number of classes
    batch_size=30 #batch size
    threshold_reeig=1e-4 #threshold for ReEig layer
    epochs=200

    if not Path(data_path).is_dir():
        print("Error: dataset not found")
        print("Please download and extract the file at the following url: http://www-connex.lip6.fr/~schwander/datasets/afew.tgz")
        sys.exit(2)

    #setup data and model
    model=AfewNet()

    #setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opti = MixOptimizer(model.parameters(),lr=lr)

    #initial validation accuracy
    loss_val,acc_val=[],[]
    y_true,y_pred=[],[]
    gen = data_loader._test_generator
    model.eval()
    for local_batch, local_labels in gen:
        out = model(local_batch)
        l = loss_fn(out, local_labels)
        predicted_labels=out.argmax(1)
        y_true.extend(list(local_labels.cpu().numpy())); y_pred.extend(list(predicted_labels.cpu().numpy()))
        acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
        loss_val.append(loss)
        acc_val.append(acc)
    acc_val = np.asarray(acc_val).mean()
    loss_val = np.asarray(loss_val).mean()
    print('Initial validation accuracy: '+str(100*acc_val)+'%')

    #training loop
    for epoch in range(epochs):

        # train one epoch
        loss_train, acc_train = [], []
        model.train()
        for local_batch, local_labels in data_loader._train_generator:
            opti.zero_grad()
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            acc, loss = (out.argmax(1)==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_train.append(loss)
            acc_train.append(acc)
            l.backward()
            opti.step()
        acc_train = np.asarray(acc_train).mean()
        loss_train = np.asarray(loss_train).mean()

        # validation
        loss_val,acc_val=[],[]
        y_true,y_pred=[],[]
        gen = data_loader._test_generator
        model.eval()
        for local_batch, local_labels in gen:
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            predicted_labels=out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy())); y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_val.append(loss)
            acc_val.append(acc)
        acc_val = np.asarray(acc_val).mean()
        loss_val = np.asarray(loss_val).mean()
        print('Val acc: ' + str(100*acc_val) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))

    print('Final validation accuracy: '+str(100*acc_val)+'%')
    return acc_val

if __name__ == "__main__":

    data_path='data/afew/'
    batch_size=30 #batch size

    class DatasetSPD(data.Dataset):
        def __init__(self, path, names):
            self._path = path
            self._names = names
        def __len__(self):
            return len(self._names)
        def __getitem__(self, item):
            x = np.load(self._path + self._names[item])[None, :, :].real
            x = th.from_numpy(x).double()
            y = int(self._names[item].split('.')[0].split('_')[-1])
            y = th.from_numpy(np.array(y)).long()
            return x,y

    class DataLoaderAFEW:
        def __init__(self,data_path,batch_size):
            path_train,path_test=data_path+'train/',data_path+'val/'
            for filenames in os.walk(path_train):
                names_train = sorted(filenames[2])
            for filenames in os.walk(path_test):
                names_test = sorted(filenames[2])
            self._train_generator=data.DataLoader(DatasetSPD(path_train,names_train),batch_size=batch_size,shuffle='True')
            self._test_generator=data.DataLoader(DatasetSPD(path_test,names_test),batch_size=batch_size,shuffle='False')

    afew(DataLoaderAFEW(data_path,batch_size))
