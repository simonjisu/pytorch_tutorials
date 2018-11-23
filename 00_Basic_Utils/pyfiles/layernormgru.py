# coding utf-8
# author: simonjisu
# github: https://github.com/simonjisu

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from itertools import accumulate


class LayerNormGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, layernorm=False,
                 bidirectional=False, bias=True, use_cuda=False, return_all_hidden=False):
        """
        Args:
            : input_size: The number of expected features in the input `x`
            : hidden_size: The number of features in the hidden state `h`
            : num_layers: Number of recurrent layers.
            : batch_first: If ``True``, then the input and output tensors are provided as `(batch, seq, feature)`
            : layernorm: If ``True``, then use torch.nn.Layernorm to normalize linear output in gru
            : bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
            : bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
            : use_cuda: If ``True``, then use cuda to init hidden state. (didn't figure out how to auto detect it, yet) Default: ``False``
            : return_all_hidden: If ``True``, return all hidden layers output. Default: ``False``
        [no packed size // packed size]
            Input:
            : inputs: tensor(seq_len, batch_size, input_size) // 'tensor(sum(batch_sizes), input_size)'
            : hidden: tensor(num_layers * num_directions, batch_size, hidden_size) if nothing then auto initialize as zeros
            output:
            : output: tensor(seq_len, batch_size, hidden_size * num_directions) // tensor(sum(batch_sizes), hidden_size * num_directions)
            : hidden: tensor(num_layers * num_directions, B, hidden_size)
        """
        super(LayerNormGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layernorm = layernorm
        self.batch_first = batch_first
        self.bidrectional = bidirectional
        self.use_cuda = use_cuda
        self.return_all_hidden = return_all_hidden
        self.num_directions = 2 if self.bidrectional else 1
        self.bias = bias
        self.gate_num = 3

    def forward(self, inputs, hidden=None):
        """
        [no packed size // packed size]
        input:
        * inputs: tensor(seq_len, batch_size, input_size) // 'tensor(sum(batch_sizes), input_size)'
        * hidden: tensor(num_layers * num_directions, batch_size, hidden_size) if nothing then auto initialize as zeros
        output:
        * output: tensor(seq_len, batch_size, hidden_size * num_directions) // tensor(sum(batch_sizes), hidden_size * num_directions)
        * hidden: tensor(num_layers * num_directions, B, hidden_size)
        """
        is_packed = isinstance(inputs, PackedSequence)
        if is_packed:
            inputs, batch_sizes = inputs
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = inputs.size(0) if self.batch_first else inputs.size(1)

        if hidden is None:
            hidden = self.init_hidden(max_batch_size)

        self.func = StackedGRU(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=self.bidrectional,
                               layernorm=self.layernorm,
                               return_all_hidden=self.return_all_hidden,
                               use_cuda=self.use_cuda,
                               is_packed=is_packed)
        if self.batch_first and not is_packed:
            inputs = inputs.transpose(0, 1)  # B, T, D --> T, B, D

        output, hidden = self.func(inputs, hidden, batch_sizes=batch_sizes)

        if self.batch_first and not is_packed:
            output = output.transpose(0, 1)
        if is_packed:
            output = PackedSequence(output, batch_sizes)

        return output, hidden

    def init_hidden(self, max_batch_size):
        hx = torch.zeros(self.num_layers * self.num_directions, max_batch_size, self.hidden_size, requires_grad=False)
        if self.use_cuda:
            hx = hx.cuda()
        return hx


class StackedGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, layernorm=False,
                 return_all_hidden=False, is_packed=False, use_cuda=False):
        super(StackedGRU, self).__init__()
        self.layernorm = layernorm
        self.bidirec = bidirectional
        self.return_all_hidden = return_all_hidden
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_directions = 2 if self.bidirec else 1
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.build_layers(input_size, hidden_size)
        # packed seq
        self.is_packed = is_packed

    def build_layers(self, input_size, hidden_size):
        self.layers = self.create_layers(input_size, hidden_size)
        if self.bidirec:
            self.r_layers = self.create_layers(input_size, hidden_size)

    def create_layers(self, input_size, hidden_size):
        layers = nn.ModuleList()
        for _ in range(self.num_layers):
            cell = GRUCell(input_size, hidden_size, layernorm=self.layernorm)
            if self.use_cuda:
                cell = cell.cuda()
            layers.append(cell)
            input_size = hidden_size
        return layers

    def forward(self, inputs, hidden, batch_sizes=None):
        """
        [no packed size // packed size]
        - D: input_size
        - B: batch_size
        - H: hidden_size
        - batch_sizes: list of batch sizes when using PackedSequence
        * input:
        inputs: 'tensor(T, B, D)' // 'tensor(sum(batch_sizes), D)'
        hidden: 'tensor(num_layers * num_directions, B, H)'
        * return:
        output: 'tensor(T, B, num_directions*H)' // tensor(sum(batch_sizes), H)
                 if return_all_hiddens
                 - 'tensor(num_layers, T, B, num_directions*H)' // 'tensor(num_layers, sum(batch_sizes), num_directions*H)'
        hidden 'tensor(num_layers*num_directions, B, H)'
        """
        if self.bidirec:
            # output (num_layers, T, B, 2H)
            # last_hidden (num_layers*num_directions, B, H)
            # forward: idx of time t ~ (0, 1, ..., T-1)
            f_idx = [i for i in range(self.num_layers * self.num_directions) if i % 2 == 0]
            f_all_outputs, f_last_hidden = self._forward(self.layers, inputs, hidden[f_idx, :], batch_sizes)

            # backward:
            r_inputs = self._flip(inputs, 0)  # (T, B, H) idx of time t ~ (T-1, ... , 0)
            b_idx = [i for i in range(self.num_layers * self.num_directions) if i % 2 != 0]
            b_all_outputs, b_last_hidden = self._forward(self.r_layers, r_inputs, hidden[b_idx, :], batch_sizes)
            b_all_outputs = self._flip(b_all_outputs, 1)  # (num_layers, T, B, H) idx of time t ~ (0, 1, ..., T-1)
            # concate layers
            # f: hidden[T-1], b: hidden[0]
            output = torch.cat([f_all_outputs, b_all_outputs], -1)
            idx = [int(i / self.num_directions) if i % 2 == 0 else \
                       i + int(((self.num_layers * self.num_directions) - i) / 2) \
                   for i in range(self.num_layers * self.num_directions)]
            hidden = torch.cat([f_last_hidden, b_last_hidden])[idx, :]

            if self.return_all_hidden:
                return output, hidden
            return output[-1], hidden

        else:
            f_all_outputs, f_last_hidden = self._forward(self.layers, inputs, hidden, batch_sizes)
            if self.return_all_hidden:
                return f_all_outputs, f_last_hidden
            return f_all_outputs[-1], f_last_hidden

    def _forward(self, layers, inputs, hidden, batch_sizes=None):
        """
        * input:
        layers: nn.ModuleList for one direction layers
        inp: tensor(T, B, D) // tensor(sum(batch_sizes), D)
        hid: num_layers, B, H (init hidden)
        * return:
        all_outputs: all layers a forward or backward layer
        tensor(num_layers, T, B, H) // tensor(num_layers, sum(batch_sizes), H)
        last_hidden:
        tensor(num_layers, B, H)
        """
        assert isinstance(layers, nn.ModuleList)
        if self.is_packed:
            assert batch_sizes is not None, 'packed sequence must have list of batch_sizes'
            acc_bs = [0] + list(accumulate(batch_sizes.tolist()))

        # all_outputs
        # if packed: num_layers, sum(batch_sizes), H
        # if not packed : num_layers, T, B, H
        all_outputs = []
        for l_idx, layer in enumerate(layers):
            hid = hidden.chunk(self.num_layers, 0)[l_idx].squeeze(0)  # init hidden: 1, B, H --> B, H
            output_ith_layer = []

            if self.is_packed:
                # packed
                for t in range(len(batch_sizes)):  # input: acc_bs[t:(t+1)]
                    hid = layer(inputs[acc_bs[t]:acc_bs[t + 1]], hid[:batch_sizes[t]])
                    output_ith_layer.append(hid)
                output_ith_layer = torch.cat(output_ith_layer, 0)  # sum(batch_sizes), H
            else:
                # not packed
                for t in range(inputs.size(0)):
                    hid = layer(inputs[t], hid)
                    output_ith_layer.append(hid)
                output_ith_layer = torch.stack(output_ith_layer)  # T, B, H

            inputs = output_ith_layer
            all_outputs.append(output_ith_layer)
        all_outputs = torch.stack(all_outputs)
        if self.is_packed:
            last_idx = self._get_last_idx(all_outputs.size(1), batch_sizes, acc_bs)
            last_hidden = torch.stack([out[last_idx] for out in all_outputs])  # num_layer, max_batch_size, H
        else:
            last_hidden = torch.stack([out[-1] for out in all_outputs])  # num_layer, B, H

        return all_outputs, last_hidden

    def _get_last_idx(self, total_len, batch_sizes, acc_bs):
        """
        there is difference output between packed rnn and not packed rnn:
        https://discuss.pytorch.org/t/lstm-hidden-cell-outputs-and-packed-sequence-for-variable-length-sequence-inputs/1183
        """
        batch_sizes = batch_sizes if isinstance(batch_sizes, list) else batch_sizes.tolist()
        mask = batch_sizes + [0]
        mask = [mask[i + 1] - mask[i] for i in range(len(batch_sizes))]
        temp = list(range(total_len))
        result = []
        for i, m in enumerate(mask):
            if m != 0:
                result.extend(temp[acc_bs[i]:acc_bs[i + 1]][m:])
        return list(reversed(result))

    def _flip(self, x, dim):
        """
        https://discuss.pytorch.org/t/optimizing-diagonal-stripe-code/17777/16
        """
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, layernorm=False, gate_num=3):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.bias = bias
        self.hidden_size = hidden_size
        self.layernorm = layernorm
        self.gate_num = gate_num

        self.weight_ih = nn.Linear(input_size, gate_num * hidden_size, bias=bias)
        self.weight_hh = nn.Linear(hidden_size, gate_num * hidden_size, bias=bias)
        if self.layernorm:
            self.lm_r = nn.LayerNorm(hidden_size)
            self.lm_i = nn.LayerNorm(hidden_size)
            self.lm_n = nn.LayerNorm(hidden_size)

    def forward(self, inputs, hidden):
        """
        inputs:
        * inputs: B, input_size
        * hidden: B, hidden_size
        output:
        * hy: B, hidden_size
        """
        gi = self.weight_ih(inputs)
        gh = self.weight_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        a_r = i_r + h_r
        a_i = i_i + h_i
        if self.layernorm:
            a_r = self.lm_r(a_r)
            a_i = self.lm_i(a_i)

        resetgate = torch.sigmoid(a_r)
        inputgate = torch.sigmoid(a_i)

        a_n = i_n + resetgate * h_n
        if self.layernorm:
            a_n = self.lm_n(a_n)

        newgate = torch.tanh(a_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy