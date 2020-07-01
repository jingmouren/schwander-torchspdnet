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

class RadarNet(nn.Module):
    def __init__(self):
        super(__class__,self).__init__()
        window_size=20; hop_length=1
        dim1=16; dim2=8
        classes=3
        self.split=nn_cplx.SplitSignal_cplx(2,window_size,hop_length)
        self.covpool=nn_cplx.CovPool_cplx()
        self.re=spdnet.ReEig()
        self.bimap1=spdnet.BiMap(1,1,window_size,dim1)
        self.bimap2=spdnet.BiMap(1,1,dim1,dim2)
        self.logeig=spdnet.LogEig()
        self.linear=nn.Linear(dim2**2,classes).double()
    def forward(self,x):
        x_split=self.split(x)
        x_spd=self.covpool(x_split)
        x_spd=self.re(self.bimap1(x_spd))
        x_spd=self.bimap2(x_spd)
        x_vec=self.logeig(x_spd).view(x_spd.shape[0],-1)
        y=self.linear(x_vec)
        return y

def radar_spdnet(data_loader):

    #main parameters
    lr=1e-2 #learning rate
    n=20 #dimension of the data
    C=3 #number of classes
    threshold_reeig=1e-4 #threshold for ReEig layer
    epochs=200

    #setup data and model
    model=RadarNet()

    #setup loss and optimizer
    loss_fn=nn.CrossEntropyLoss()
    opti=MixOptimizer(model.parameters(),lr=lr,momentum=.9,weight_decay=5e-4)

    #training loop
    best=0
    for epoch in range(epochs):

        # validation
        loss_val,acc_val=[],[]
        y_true,y_pred=[],[]
        gen = data_loader._val_generator
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
        print('Val acc: ' + str(100*acc_val) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))

    #test accuracy
    loss_test,acc_test=[],[]
    y_true,y_pred=[],[]
    gen = data_loader._test_generator
    model.eval()
    for local_batch, local_labels in gen:
        out = model(local_batch)
        l = loss_fn(out, local_labels)
        predicted_labels=out.argmax(1)
        y_true.extend(list(local_labels.cpu().numpy())); y_pred.extend(list(predicted_labels.cpu().numpy()))
        acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
        loss_test.append(loss)
        acc_test.append(acc)
    acc_test = np.asarray(acc_test).mean()
    loss_test = np.asarray(loss_test).mean()
    print('Final validation accuracy: '+str(100*acc_val)+'%')
    print('Test accuracy: '+str(100*acc_test)+'%')
    return 100*acc_val


if __name__ == "__main__":

    data_path='data/radar/' #data path
    pval=0.25 #validation percentage
    ptest=0.25 #test percentage
    batch_size=30 #batch size

    class DatasetRadar(data.Dataset):
        def __init__(self, path, names):
            self._path = path
            self._names = names
        def __len__(self):
            return len(self._names)
        def __getitem__(self, item):
            x=np.load(self._path+self._names[item])
            x=np.concatenate((x.real[:,None],x.imag[:,None]),axis=1).T
            x=th.from_numpy(x)
            y=int(self._names[item].split('.')[0].split('_')[-1])
            y=th.from_numpy(np.array(y))
            return x.float(),y.long()
    class DataLoaderRadar:
        def __init__(self,data_path,pval,batch_size):
            for filenames in os.walk(data_path):
                names=sorted(filenames[2])
            random.Random(0).shuffle(names)
            N_val=int(pval*len(names))
            N_test=int(ptest*len(names))
            N_train=len(names)-N_test-N_val
            train_set=DatasetRadar(data_path,names[N_val+N_test:int(N_train)+N_test+N_val])
            test_set=DatasetRadar(data_path,names[:N_test])
            val_set=DatasetRadar(data_path,names[N_test:N_test+N_val])
            self._train_generator=data.DataLoader(train_set,batch_size=batch_size,shuffle='True')
            self._test_generator=data.DataLoader(test_set,batch_size=batch_size,shuffle='False')
            self._val_generator=data.DataLoader(val_set,batch_size=batch_size,shuffle='False')

    radar_spdnet(DataLoaderRadar(data_path,pval,batch_size))
