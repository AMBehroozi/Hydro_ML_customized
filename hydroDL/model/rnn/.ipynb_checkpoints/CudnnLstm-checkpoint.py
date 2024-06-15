# """A class to hold the CudnnLSTM layer"""
# import math
# import torch
# import torch.nn as nn
# from torch.nn import Parameter
# import torch.nn.functional as F
# from hydroDL.model.dropout import DropMask, createMask
# import csv
# import numpy as np
# import torch.fft


# class CudnnLstm(nn.Module):
#     def __init__(self, *, inputSize, hiddenSize, nmodes, dr=0.5, drMethod="drW", gpu=0, seed=42):
#         super(CudnnLstm, self).__init__()
#         self.inputSize = inputSize
        
#         self.hiddenSize = hiddenSize
#         self.dr = dr
#         self.nmodes = nmodes
#         self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
#         self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
#         self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
#         self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
#         self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
#         self.cuda()
#         self.name = "CudnnLstm"
#         self.seed = seed
#         self.is_legacy = True

#         self.reset_mask()
#         self.reset_parameters()

#     def _apply(self, fn):
#         ret = super(CudnnLstm, self)._apply(fn)
#         return ret

#     def __setstate__(self, d):
#         super(CudnnLstm, self).__setstate__(d)
#         self.__dict__.setdefault("_data_ptrs", [])
#         if "all_weights" in d:
#             self._all_weights = d["all_weights"]
#         if isinstance(self._all_weights[0][0], str):
#             return
#         self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

#     def reset_mask(self):
#         self.maskW_ih = createMask(self.w_ih, self.dr, self.seed)
#         self.maskW_hh = createMask(self.w_hh, self.dr, self.seed)

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hiddenSize)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)

#     def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        
#         fft_input = torch.fft.fft(input, dim=0)

#         # Keep only the first three Fourier modes
#         m = self.nmodes
#         fft_input[:, m:, :] = 0  # Zero out all modes after the first three

#         # Apply inverse Fourier transform
#         ifft_input = torch.fft.ifft(fft_input, dim=0).real

        
#         # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
#         if dropoutFalse and (not doDropMC):
#             doDrop = False
#         elif self.dr > 0 and (doDropMC is True or self.training is True):
#             doDrop = True
#         else:
#             doDrop = False

#         batchSize = input.size(1)

#         if hx is None:
#             hx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
#         if cx is None:
#             cx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)

#         # cuDNN backend - disabled flat weight
#         # handle = torch.backends.cudnn.get_handle()
#         if doDrop is True:
#             self.reset_mask()
#             weight = [
#                 DropMask.apply(self.w_ih, self.maskW_ih, True),
#                 DropMask.apply(self.w_hh, self.maskW_hh, True),
#                 self.b_ih,
#                 self.b_hh,
#             ]
#         else:
#             weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

#         # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
#         # input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
#         # self.hiddenSize, 1, False, 0, self.training, False, (), None)
#         if torch.__version__ < "1.8":
#             output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
#                 input,
#                 weight,
#                 4,
#                 None,
#                 hx,
#                 cx,
#                 2,  # 2 means LSTM
#                 self.hiddenSize,
#                 1,
#                 False,
#                 0,
#                 self.training,
#                 False,
#                 (),
#                 None,
#             )
#         else:
#             output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
#                 input,
#                 weight,
#                 4,
#                 None,
#                 hx,
#                 cx,
#                 2,  # 2 means LSTM
#                 self.hiddenSize,
#                 0,
#                 1,
#                 False,
#                 0,
#                 self.training,
#                 False,
#                 (),
#                 None,
#             )
#         return output, (hy, cy)

#     @property
#     def all_weights(self):
#         return [
#             [getattr(self, weight) for weight in weights]
#             for weights in self._all_weights
#         ]







# """A class to hold the CudnnLSTM layer"""
# import math
# import torch
# import torch.nn as nn
# from torch.nn import Parameter
# import torch.nn.functional as F
# from hydroDL.model.dropout import DropMask, createMask
# import csv
# import numpy as np
# import torch.fft


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FrequencyFilter(nn.Module):
#     def __init__(self, sequence_length, nmodes):
#         super(FrequencyFilter, self).__init__()
#         # Create a mask for the first 'nmodes' frequencies
#         self.mask = torch.zeros(sequence_length)
#         self.mask[:nmodes] = 1
#         # Make it a parameter to learn which modes to keep, but without training it
#         self.mask = nn.Parameter(self.mask, requires_grad=False)

#     def forward(self, input):
#         # Assuming input is of shape [batch, sequence_length], apply FFT
#         input_fft = torch.fft.fft(input, dim=1)
#         # Apply the mask to the FFT transformed data
#         filtered_fft = input_fft * self.mask
#         # Convert back using IFFT
#         filtered_signal = torch.fft.ifft(filtered_fft, dim=1).real
#         return filtered_signal



# class CudnnLstm(nn.Module):
#     def __init__(self, *, inputSize, hiddenSize, nmodes, dr=0.5, drMethod="drW", gpu=0, seed=42):
#         super(CudnnLstm, self).__init__()
#         self.inputSize = inputSize
        
#         self.hiddenSize = hiddenSize
#         self.dr = dr
#         self.sequence_filter = FrequencyFilter(sequence_length=inputSize, nmodes=nmodes)
#         self.nmodes = nmodes
#         self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
#         self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
#         self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
#         self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
#         self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
#         self.cuda()
#         self.name = "CudnnLstm"
#         self.seed = seed
#         self.is_legacy = True

#         self.reset_mask()
#         self.reset_parameters()

#     def _apply(self, fn):
#         ret = super(CudnnLstm, self)._apply(fn)
#         return ret

#     def __setstate__(self, d):
#         super(CudnnLstm, self).__setstate__(d)
#         self.__dict__.setdefault("_data_ptrs", [])
#         if "all_weights" in d:
#             self._all_weights = d["all_weights"]
#         if isinstance(self._all_weights[0][0], str):
#             return
#         self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

#     def reset_mask(self):
#         self.maskW_ih = createMask(self.w_ih, self.dr, self.seed)
#         self.maskW_hh = createMask(self.w_hh, self.dr, self.seed)

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.hiddenSize)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)

#     def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        
#         x_filtered = self.sequence_filter(input)


#         ifft_input = x_filtered


        
#         # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
#         if dropoutFalse and (not doDropMC):
#             doDrop = False
#         elif self.dr > 0 and (doDropMC is True or self.training is True):
#             doDrop = True
#         else:
#             doDrop = False

#         batchSize = input.size(1)

#         if hx is None:
#             hx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
#         if cx is None:
#             cx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)

#         # cuDNN backend - disabled flat weight
#         # handle = torch.backends.cudnn.get_handle()
#         if doDrop is True:
#             self.reset_mask()
#             weight = [
#                 DropMask.apply(self.w_ih, self.maskW_ih, True),
#                 DropMask.apply(self.w_hh, self.maskW_hh, True),
#                 self.b_ih,
#                 self.b_hh,
#             ]
#         else:
#             weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

#         # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
#         # input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
#         # self.hiddenSize, 1, False, 0, self.training, False, (), None)
#         if torch.__version__ < "1.8":
#             output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
#                 input,
#                 weight,
#                 4,
#                 None,
#                 hx,
#                 cx,
#                 2,  # 2 means LSTM
#                 self.hiddenSize,
#                 1,
#                 False,
#                 0,
#                 self.training,
#                 False,
#                 (),
#                 None,
#             )
#         else:
#             output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
#                 input,
#                 weight,
#                 4,
#                 None,
#                 hx,
#                 cx,
#                 2,  # 2 means LSTM
#                 self.hiddenSize,
#                 0,
#                 1,
#                 False,
#                 0,
#                 self.training,
#                 False,
#                 (),
#                 None,
#             )
#         return output, (hy, cy)

#     @property
#     def all_weights(self):
#         return [
#             [getattr(self, weight) for weight in weights]
#             for weights in self._all_weights
#         ]


"""A class to hold the CudnnLSTM layer"""
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
import csv
import numpy as np


class CudnnLstm(nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod="drW", gpu=0, seed=42):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
        self.cuda()
        self.name = "CudnnLstm"
        self.seed = seed
        self.is_legacy = True

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault("_data_ptrs", [])
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr, self.seed)
        self.maskW_hh = createMask(self.w_hh, self.dr, self.seed)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        # input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        # self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hiddenSize,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hiddenSize,
                0,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]
