import warnings

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


# Cell code from https://pytorch.org/docs/stable/nn.html#torch.nn.RNNCell
# Parameters and initializations based on https://github.com/batzner/indrnn/blob/master/ind_rnn_cell.py

class IndRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, activation="relu",
                 recurrent_min_abs=None, recurrent_max_abs=None, hidden_initializer=None,
                 recurrent_initializer=None, gradient_clip_min=None, gradient_clip_max=None):
        super(IndRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(self.hidden_size, self.input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            warnings.warn("IndRNN supports only ReLu and tanh activations. Fallingback to ReLU ")
            self.activation = F.relu
        self.recurrent_min_abs = recurrent_min_abs
        self.recurrent_max_abs = recurrent_max_abs
        self.hidden_initializer = hidden_initializer
        self.recurrent_initializer = recurrent_initializer

        # Gradient Clippnig to prevent Gradient Explosion and over fitting
        if not gradient_clip_max is None:
            self.gradient_clip_min = -gradient_clip_max
            self.gradient_clip_max = gradient_clip_max
            if not gradient_clip_min is None:
                self.gradient_clip_min = gradient_clip_min
            # register_hook will record the change to the parameter made
            # into the grad and this will be used during gradient descent
            self.weight_ih.register_hook(lambda x: x.clamp_(min=gradient_clip_min, max=gradient_clip_max))
            self.weight_hh.register_hook(lambda x: x.clamp_(min=gradient_clip_min, max=gradient_clip_max))
            if self.bias:
                self.bias_ih.register_hook(lambda x: x.clamp_(min=gradient_clip_min, max=gradient_clip_max))

        # Initialize all parametere of the model
        for name, weight in self.named_parameters():
            if "bias" in name:
                # self.add_variable("bias", shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
                weight.data.zero_()
            elif "weight_ih" in name:
                # self._input_initializer = init_ops.random_normal_initializer(mean=0.0, stddev=0.001)
                if self.hidden_initializer is None:
                    nn.init.normal_(weight, 0, 0.01)
                else:
                    self.hidden_initializer(weight)
            elif "weight_hh" in name:
                # self._recurrent_initializer = init_ops.constant_initializer(1.)
                if self.recurrent_initializer is None:
                    nn.init.constant_(weight, 1)
                else:
                    self.recurrent_initializer(weight)
            else:
                weight.data.normal_(0, 0.01)
        self.clip_recurrent_weights()

    def clip_recurrent_weights(self):
        # Clip the absolute values of the recurrent weights to the specified minimum
        r"""
        Code from https://github.com/batzner/indrnn/blob/master/ind_rnn_cell.py
        # Clip the absolute values of the recurrent weights to the specified minimum
            if self._recurrent_min_abs:
              abs_kernel = math_ops.abs(self._recurrent_kernel)
              min_abs_kernel = math_ops.maximum(abs_kernel, self._recurrent_min_abs)
              self._recurrent_kernel = math_ops.multiply(
                  math_ops.sign(self._recurrent_kernel),
                  min_abs_kernel
              )

            # Clip the absolute values of the recurrent weights to the specified maximum
            if self._recurrent_max_abs:
              self._recurrent_kernel = clip_ops.clip_by_value(self._recurrent_kernel,
                                                              -self._recurrent_max_abs,
                                                              self._recurrent_max_abs)
        """
        if self.recurrent_min_abs:
            abs_kernel = torch.abs(self.weight_hh.data).clamp_(min=self.recurrent_min_abs)
            self.weight_hh.data = abs_kernel.mm(torch.sign(self.weight_hh.data))
        if self.recurrent_max_abs:
            self.weight_hh.data = self.weight_hh.clamp(max=self.recurrent_max_abs, min=-self.recurrent_max_abs)

        # if self.recurrent_min_abs:
        #     # abs_kernel = torch.abs(self.weight_hh.data).clamp_(min=self.recurrent_min_abs)
        #     # self.weight_hh.data = self.weight_hh.mul(torch.sign(self.weight_hh.data), abs_kernel)
        #     abs_kernel = torch.abs(self.weight_hh.data).clamp_(min=self.recurrent_min_abs)
        #     self.weight_hh.data = self.weight_hh.mul(torch.sign(self.weight_hh.data), abs_kernel)
        #
        # # Clip the absolute values of the recurrent weights to the specified maximum
        # if self.recurrent_max_abs:
        #     self.weight_hh.data = self.weight_hh.clamp(min=-self._recurrent_max_abs,
        #                                                max=self._recurrent_max_abs)

        # Pendnng: Implement code for dropouts
        # --------

    def forward(self, input, hx=None):
        # out = tanh(w_{ih} * x + b_{ih}  +  w_{hh} (*) h)
        # (*) Hammard Product
        return self.activation(F.linear(input , self.weight_ih , self.bias_ih) + F.mul(self.weight_hh, hx))


class IndRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 hidden_initializers=None, recurrent_initializers=None,
                 batch_normalizer=None, bidirectional=False, **kwargs):
        super(IndRNN,self).__init__()
        self.input_size = input_size
        self.hidden_initializers = hidden_initializers
        self.hidden_size = hidden_size
        self.recurrent_initializers = recurrent_initializers
        self.num_layers = num_layers
        self.batch_normalizer = batch_normalizer
        # Logic for bidirectional pending
        # Refer https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.batch_index = 1
        self.time_index = 0

        cells_tmp = []
        batch_norms_tmp = []
        for i in range(num_layers):
            if recurrent_initializers is not None:
                kwargs['recurrent_initializer'] = recurrent_initializers[i]
            if hidden_initializers is not None:
                kwargs['hidden_initializer'] = hidden_initializers[i]
            if i == 0:
                cells_tmp.append(IndRNNCell(self.input_size, self.hidden_size, **kwargs))
            else:
                cells_tmp.append(IndRNNCell(self.hidden_size, self.hidden_size, **kwargs))
            if batch_normalizer:
                batch_norms_tmp.append(nn.BatchNorm1d(self.hidden_size))
        self.cells = nn.ModuleList(cells_tmp)
        self.batch_norms = nn.ModuleList(batch_norms_tmp)

        h_tmp = torch.zeros(hidden_size)
        self.register_buffer('h_tmp', torch.autograd.Variable(h_tmp))

    def forward(self, input, hidden):
        for i, cell in enumerate(self.cells):
            # here the h_tmp tensor of zeros is expanded to a size (no_of_batches, hidden size)
            hx = self.h_tmp.unsqueeze(0).expand(input.size(self.batch_index) , self.hidden_size).contiguous()
            cell.clip_recurrent_weights()

            output = []
            X = torch.unbind(input, self.time_index)
            for x in X:
                hx = cell(x, hx)
                # if self.batch_normalizer:
                #     hx = self.bns[i](hx)
                output.append(hx)
            input = torch.stack(output, self.time_index)
            if self.batch_normalizer:
                input = self.batch_norms[i](input)
        return input.squeeze(2)