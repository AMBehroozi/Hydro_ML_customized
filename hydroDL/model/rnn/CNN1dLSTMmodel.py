import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
from hydroDL.model import cnn, rnn
import csv
import numpy as np


class CNN1dLSTMmodel(torch.nn.Module):
    def __init__(
        self,
        *,
        nx,
        ny,
        nobs,
        hiddenSize,
        nkernel=(10, 5),
        kernelSize=(3, 3),
        stride=(2, 1),
        dr=0.5,
        poolOpt=None
    ):
        # two convolutional layer
        super(CNN1dLSTMmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = cnn.CNN1dkernel(
                ninchannel=ninchan,
                nkernel=nkernel[ii],
                kernelSize=kernelSize[ii],
                stride=stride[ii],
            )
            self.features.add_module("CnnLayer%d" % (ii + 1), ConvLayer)
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module("Relu%d" % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module(
                    "Pooling%d" % (ii + 1), nn.MaxPool1d(poolOpt[ii])
                )
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(
            Lout * nkernel[-1]
        )  # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = rnn.CudnnLstm(inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.name = "CNN1dLSTMmodel"
        self.is_legacy = True

    def forward(self, x, z, doDropMC=False):
        nt, ngrid, nobs = z.shape
        z = z.view(nt * ngrid, 1, nobs)
        z0 = self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0 = z0.view(nt, ngrid, self.Ncnnout)
        x0 = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x0))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out
