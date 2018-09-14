import warnings

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

# Cell code from https://pytorch.org/docs/stable/nn.html#torch.nn.RNNCell
# Parameters and initializations based on https://github.com/batzner/indrnn/blob/master/ind_rnn_cell.py

class IndRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, activation="relu",
                 recurrent_min_abs=None, recurrent_max_abs=None, hidden_initializer = None,
                 recurrent_initializer=None, gradient_clip_min = 0, gradient_clip_max = 0):
        super(IndRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
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
            self.weight_hh.data = self.weight_hh.mul(torch.sign(self.weight_hh), abs_kernel)

        # Clip the absolute values of the recurrent weights to the specified maximum
        if self.recurrent_max_abs:
            self.weight_hh.data = self.weight_hh.clamp(min = -self._recurrent_max_abs,
                                                    max = self._recurrent_max_abs)

        # Implement code for dropouts

    def forward(self, input, hx=None):
        # out = tanh(w_{ih} * x + b_{ih}  +  w_{hh} (*) h)
        # (*) Hammard Product
        return self.activation((self.weight_ih * input + self.bias_ih) + (torch.mul(self.weight_hh, hx)))