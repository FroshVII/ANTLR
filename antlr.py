"""Spiking Neural Network (SNN) from Kim et al, 2020."""

from torch.autograd import Variable, Function
from torch.nn import init
from torch.nn.parameter import Parameter

import gc  # garbage colletor
import logging
import matplotlib.pyplot as plt
import math
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.sgd

# ======= ======= =======
# Utilities
# ======= ======= ====

def is_conv2d_or_linear(layer):
    """
    Return `True` if `layer` is an instance of either `nn.Conv2d` or
    `nn.Linear`, else `False`.
    """
    is_conv2d = isinstance(layer, nn.Conv2d)
    return is_conv2d or isinstance(layer, nn.Linear)

def is_apool_or_flatten(layer):
    """
    Return `True` if `layer` is an instance of either `nn.AvgPool2d` or
    `nn.Flatten`, else `False`.
    """
    is_apool = isinstance(layer, nn.AvgPool2d)
    return is_apool or isinstance(layer, nn.Flatten)

def is_nonetype(obj):
    """`True` if an object has the same type as `None` else `False`."""

    # Implemented here exactly the way the original authors implemented
    # it, because I was unsure what reason they had to check the two
    # objects' types instead of using `is None` or `NoneType` directly,
    # but assumed it was done to deal with some edge behaviors. If that
    # is not the case, I recommend replacing occurences in the code of
    # `is_nonetype(x)` with the Pythonic `x is None` where possible.
    return type(obj) == type(None)

def clamp_grad(obj, grad_clip):
    """Clamp gradients to plus-or-minus the clip value.

    Clamps an object's gradients to the range
    [-|grad_clip|, +|grad_clip|]. Object must have attribute `grad`.
    """
    abs_gc = abs(grad_clip)
    obj.grad = torch.clamp(obj.grad, -abs_gc, abs_gc)

# ======= ======= =======
# Model Definition
# ======= ======= ====

class ListSNNMulti(nn.Module):
    """Simple spiking neural network (SNN) model without SNNCell"""

    # ======= ======= =======
    # Class Variables
    # ======= ======= ====

    # Loss targets supported by the model implementation
    VALID_TARGETS = ["count", "train", "latency"]

    # ======= ======= =======
    # Utilities
    # ======= ======= ====

    def no_mm_support(self, layer_type):
        """
        Throw an exception alerting the user a layer type they tried to
        use is not supported for multi-models.
        """
        if self.multi_model:
            raise NotImplementedError(
                f"layer type '{layer_type}' currently unsupported for " +
                "multi-models"
            )

    def exists_and_true(self, attr_key):
        """Return the value of an attribute if it's set, else False."""
        if hasattr(self, attr_key):
            attr_val = getattr(self, attr_key)
            if type(attr_val) != bool:
                logging.warning(f"{attr_key} is not a boolean")
            return attr_val
        else:
            return False

    # ======= ======= =======
    # Initialization
    # ======= ======= ====

    def _init_kernels(self):
        """Initialize kernels as attributes."""

        # Make sure a valid target type is selected
        if self.target_type not in self.VALID_TARGETS:
            raise ValueError(
                f"invalid target type '{self.target_type}', "
                + f"valid types are {self.VALID_TARGETS}"
            )

        # Code offers support for three different types of loss
        # function and corresponding activation-based gradient (count,
        # spike-train, and latency). See Table 1 in Kim et al, 2020 for
        # more details.
        if self.target_type in ["train", "count"]:

            if self.target_type == "train":
                # self.alpha_exp = 0.9
                self.alpha_extend = 200
            elif self.target_type == "count":
                self.alpha_exp = 1.0
                self.alpha_extend = 0

            kernel = torch.pow(
                self.alpha_exp,
                torch.arange(min(self.time_length, 1000)).float()
            )
            kernel = kernel[kernel > 1e-06].view(1, 1, -1)

            kernel_shifted_front = torch.cat((kernel, torch.zeros(1, 1, 2)), 2)
            kernel_shifted_back = torch.cat((torch.zeros(1, 1, 2), kernel), 2)
            kernel_prime = (kernel_shifted_front - kernel_shifted_back) / 2

            self.alpha_kernel = kernel
            self.alpha_kernel_prime = kernel_prime

        # For calculating double-exponential kernel, we calculate each
        # timestep's synaptic current (exponentially decays toward future) and
        # accumulate the decayed effect (exponentially decays toward past) of
        # those synaptic currents to the current timestep.
        epsilon = torch.zeros(self.time_length)  # torch.zeros(1000)
        for t in range(epsilon.numel()):
            current_trace = torch.pow(
                self.alpha_i,
                torch.arange(t + 1).float()
            )
            trace_weight = torch.pow(
                self.alpha_v,
                torch.arange(t, -1, -1).float()
            )
            epsilon[t] = (current_trace * trace_weight).sum()
        # epsilon = epsilon[epsilon.abs() > 1e-06]

        if self.beta_auto:
            self.beta_i = 1.0
            self.beta_v = 1.0 / epsilon.max()
            self.beta_bias = 1.0 / epsilon.max()
            print(f"calculated beta_v : {self.beta_v}")
            print(f"calculated beta_bias : {self.beta_bias}")

        epsilon *= self.beta_i * self.beta_v

        epsilon_shifted_front = torch.cat((epsilon, torch.zeros(2)))
        epsilon_shifted_back = torch.cat((torch.zeros(2), epsilon))
        epsilon_prime = (epsilon_shifted_front - epsilon_shifted_back) / 2

        self.epsilon = epsilon
        self.epsilon_prime = epsilon_prime

    def _init_layers(self):
        self.num_layer = np.size(self.network_size) - 1

        self.state_v_bs = list()
        self.layers = list()
        self.fmap_shape_list = list()
        self.fmap_type_list = list()

        # Parse shape
        if "x" in self.network_size[0]:
            # Format A: string
            in_channels, height, width = [
                int(item) for item in self.network_size[0].split("x")
            ]
        else:
            # Format B: iterable
            in_channels = int(self.network_size[0])

        def _init_single_layer(layer_spec):
            """Initialize a single layer from a specification."""

            # Build convolutional layer according to specification
            if "conv" in layer_spec:
                self.no_mm_support("conv")  # unsupported for multi-models
                out_channels, kernel_size = [
                    int(item) for item in layer_spec.strip("conv").split("c")
                ]

                # layer
                padding = math.floor(kernel_size / 2)
                layer = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=padding, bias=False
                )

                # bias
                bias = Parameter(torch.Tensor(out_channels))

                # fmap
                in_channels = out_channels  # height/width unchanged
                fmap_shape = [in_channels, height, width]
                fmap_type = "conv"

            # Build fully connected layer according to specification
            elif "fc" in layer_spec:
                out_channels = int(layer_spec.strip("fc"))

                # layer
                layer = torch.nn.Linear(in_channels, out_channels, bias=False)

                # bias
                bias = Parameter(torch.Tensor(out_channels))

                # fmap
                in_channels = out_channels
                fmap_shape = [in_channels]
                fmap_type = "fc"

            # Build average-pool layer according to specification
            elif "apool" in layer_spec:
                self.no_mm_support("apool")  # unsupported for multi-models
                pool_size = int(layer_spec.strip("apool"))

                # layer
                layer = torch.nn.AvgPool2d(pool_size)

                # bias
                bias = None

                # fmap
                height = math.floor(height / pool_size)
                width = math.floor(width / pool_size)
                in_channels = out_channels
                fmap_shape = [in_channels, height, width]
                fmap_type = "apool"

            # Build max-pool layer according to specification
            elif "mpool" in layer_spec:
                self.no_mm_support("mpool")  # unsupported for multi-models
                pool_size = int(layer_spec.strip("mpool"))

                # layer
                layer = torch.nn.MaxPool2d(pool_size, return_indices=True)
                layer.max_index_list = []

                # bias
                bias = None

                # fmap
                height = math.floor(height / pool_size)
                width = math.floor(width / pool_size)
                in_channels = out_channels
                fmap_shape = [in_channels, height, width]
                fmap_type = "mpool"

            # Build flattening layer according to specification
            elif "flatten" in layer_spec:
                layer = torch.nn.Flatten()
                in_channels = in_channels * height * width
                bias = None
                height = 1
                width = 1
                fmap_shape = [in_channels]
                fmap_type = "flatten"

            # Report invalid/unsupported layer types
            else:
                raise ValueError(
                    "invalid layer type in the following specification:\n"
                    + f"{layer_spec}"
                )

            return layer, bias, fmap_shape, fmap_type

        if self.multi_model:
            # Initialize all model layers in a multi-model
            for m in range(self.num_models):
                # Initialize layers for a single model in a multi-model
                for l, layer_spec in enumerate(self.network_size[1:]):
                    # Initialize layer
                    layer, bias, fmap_shape, fmap_type = _init_single_layer(
                        layer_spec
                    )
                    # Bookkeep
                    if m == 0:
                        self.layers.append([layer])
                        self.state_v_bs.append([bias])
                        self.fmap_shape_list.append(fmap_shape)
                        self.fmap_type_list.append(fmap_type)
                    else:
                        self.layers[l].append(layer)
                        self.state_v_bs[l].append(bias)

                setattr(  # make each layer an attribute
                    self,
                    f"m{m}_layers_module",
                    nn.ModuleList([layers[m] for layers in self.layers]),
                )
                setattr(  # make all state_v_bs parameters attributes
                    self,
                    f"m{m}_state_v_bs_param",
                    nn.ParameterList([v_b[m] for v_b in self.state_v_bs]),
                )
        else:
            # Initialize all layers in a single model
            for layer_spec in self.network_size[1:]:
                # Initialize layer
                layer, bias, fmap_shape, fmap_type = _init_single_layer(layer_spec)
                # Bookkeep
                self.layers.append(layer)
                self.state_v_bs.append(bias)
                self.fmap_shape_list.append(fmap_shape)
                self.fmap_type_list.append(fmap_type)

            # Make layers and state_v_bs parameters model attributes
            self.layers_module = nn.ModuleList(self.layers)
            self.state_v_bs_param = nn.ParameterList(self.state_v_bs)

        self.reset_parameters()

    def __init__(self, model_config):
        """Instantiate model."""
        super(ListSNNMulti, self).__init__()
        print("SNNmodel_parallel instantiated")
        if hasattr(model_config, "__dict__"):
            self.__dict__.update(model_config.__dict__)
        else:
            self.__dict__.update(model_config)
        if not self.multi_model and self.num_models != 1:
            raise ValueError(
                "must set multi-model flag as True when num_models > 1"
            )
        self._init_kernels()
        self._init_layers()

    # ======= ======= =======
    # Forward Pass
    # ======= ======= ====

    def _init_param_grads(self):
        """Initialize weight and bias gradients."""
        self.weight_grad = list()
        self.bias_grad = list()
        for l, layers in enumerate(self.layers):
            # Make sure layer selection is appropriate to model type
            layer = layers[0] if self.multi_model else layers

            # Layers which are fully connected or convolutional have
            # weight and bias gradients
            if self.fmap_type_list[l] in ["fc", "conv"]:
                # Weight and bias gradients are initialized to zero
                if self.multi_model:
                    wshape = [self.num_models, *layer.weight.size()]
                    bshape = [self.num_models, *self.fmap_shape_list[l]]
                else:
                    wshape = layer.weight.size()
                    bshape = self.fmap_shape_list[l]

                zeros_w = torch.zeros(wshape, requires_grad=False)
                zeros_b = torch.zeros(bshape)

                self.weight_grad.append(zeros_w)
                self.bias_grad.append(zeros_b)

            # Layers which are not fully connected or convolutional
            # lack weight and bias gradients
            else:
                self.weight_grad.append(None)
                self.bias_grad.append(None)

    def reset_parameters(self):
        """Reset all layers' parameters."""

        def reset_parameters_single_layer(layer, state_v_bs):
            """Reset an individual layer's parameters."""

            # The following operations are only valid for linear and/or
            # convolutional layers
            if not is_conv2d_or_linear(layer, state_v_bs):
                return

            init.constant_(state_v_bs, 0)

            # Perform normal weight initialization if enabled
            if self.exists_and_true("normal_weight_init"):
                init.normal_(layer.weight, mean=0, std=self.weight_init_std)

            # Add weight bias if specified
            if hasattr(self, "weight_bias"):
                layer.weight = nn.Parameter(layer.weight + self.weight_bias)

        # Reset all layers' parameters
        for l, layer in enumerate(self.layers):
            if self.multi_model:
                # If it's a multi-model, map the reset procedure over
                # each layer
                for m in range(self.num_models):
                    reset_parameters_single_layer(layer[m], self.state_v_bs[l][m])
            else:
                # If it's a single-layer model, simply perform the reset
                # procedure on that layer
                reset_parameters_single_layer(layer, self.state_v_bs[l])

    def forward(self, input):
        """Forward pass."""

        # input: batch x time x neuron
        # state_*: layer x time x batch x neuron
        # output: batch x time x neuron
        with torch.no_grad():
            if self.multi_model:
                assert self.num_models == input.shape[0]
                self.input = input.reshape(
                    input.shape[0] * input.shape[1],
                    *input.shape[2:]
                )
            else:
                self.input = input

            if self.target_type == "latency":
                if self.multi_model:
                    dim1 = self.num_models * input.shape[1]
                else:
                    dim1 = input.shape[0]
                dim2 = self.fmap_shape_list[-1][0]
                self.output_s_cum = torch.zeros(dim1, dim2)

            self.state_i = list()
            self.state_v = list()
            self.state_v_prime = list()
            self.state_s = list()
            for l, layers in enumerate(self.layers):
                self.state_i.append(list())
                self.state_v.append(list())
                self.state_v_prime.append(list())
                self.state_s.append(list())

            # flush the max_index_list
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    layer.max_index_list = []

            # Merge parameters
            if self.multi_model:
                # Compile weights
                self.weight_list = []
                for layer_list in self.layers:
                    if hasattr(layer_list[0], "weight"):
                        weight = torch.stack(
                            [layer.weight for layer in layer_list]
                        )
                    else:
                        weight = None
                    self.weight_list.append(weight)

                # Compile biases
                self.bias_list = []
                for state_v_b_list in self.state_v_bs:
                    if state_v_b_list[0] is not None:
                        bias = torch.stack(
                            [state_v_b for state_v_b in state_v_b_list]
                        ).unsqueeze(1)
                    else:
                        bias = None
                    self.bias_list.append(bias)

            for t in range(self.time_length):
                for l, layers in enumerate(self.layers):
                    layer = layers[0] if self.multi_model else layers

                    # Convolutional or linear
                    if is_conv2d_or_linear(layer):
                        # Eq. (2)
                        if l == 0:
                            if self.multi_model:
                                state_i = torch.bmm(
                                    input[:, :, t], self.weight_list[l].permute(0, 2, 1)
                                )
                                self.state_i[l].append(
                                    state_i.reshape(
                                        state_i.shape[0] * state_i.shape[1],
                                        *state_i.shape[2:],
                                    )
                                    * self.beta_i
                                )
                            else:
                                self.state_i[l].append(layer(input[:, t]) * self.beta_i)
                        elif self.multi_model:
                            state_s = self.state_s[l - 1][-1]
                            state_s_rs = state_s.reshape(
                                self.num_models,
                                int(state_s.shape[0] / self.num_models),
                                *state_s.shape[1:],
                            )
                            state_i = (
                                torch.bmm(
                                    state_s_rs, self.weight_list[l].permute(0, 2, 1)
                                )
                                * self.beta_i
                            )
                            self.state_i[l].append(
                                state_i.reshape(
                                    state_i.shape[0] * state_i.shape[1],
                                    *state_i.shape[2:],
                                )
                            )
                        else:
                            self.state_i[l].append(
                                layer(self.state_s[l - 1][-1]) * self.beta_i
                            )
                        if t != 0:
                            idelta = self.alpha_i * self.state_i[l][t - 1]
                            idelta *= (1 - self.state_s[l][-1])
                            self.state_i[l][-1] += idelta

                        # Eq. (1)
                        if self.multi_model:
                            state_i = self.state_i[l][-1]
                            state_i_rs = state_i.reshape(
                                self.num_models,
                                int(state_i.shape[0] / self.num_models),
                                *state_i.shape[1:],
                            )
                            state_v = state_i_rs * self.beta_v
                            state_v += self.bias_list[l] * self.beta_bias
                            self.state_v[l].append(
                                state_v.reshape(
                                    state_v.shape[0] * state_v.shape[1],
                                    *state_v.shape[2:],
                                )
                            )

                        else:
                            rank = len(self.state_i[l][-1].shape)
                            if rank == 2:
                                state_v = self.state_v_bs[l]
                            elif rank == 4:
                                state_v = self.state_v_bs[l].view(-1, 1, 1)
                            else:
                                raise ValueError(
                                    "expected multi-model or model with " +
                                    "rank(I) in {2, 4}, found singular " +
                                    f"model with rank(I) of {rank}"
                                )
                            state_v *= self.beta_bias
                            state_v += self.state_i[l][-1] * self.beta_v
                            self.state_v[l].append(state_v)

                        if t != 0:
                            self.state_v[l][-1] += (
                                self.state_v[l][t - 1]
                                * (1 - self.state_s[l][-1])
                                * self.alpha_v
                            )

                            state_v_prime = (-1) * self.state_v[l][-2]
                            state_v_prime *= (1 - self.state_s[l][-1])
                            state_v_prime += self.state_v[l][-1]

                            self.state_v_prime[l].append(state_v_prime)
                        else:
                            self.state_v_prime[l].append(self.state_v[l][-1])

                        self.state_v_prime[l][-1] = torch.clamp(
                            self.state_v_prime[l][-1], min=1e-2
                        )
                        self.state_s[l].append(self.act(self.state_v[l][-1]))

                    # Average pooling or flattening
                    elif is_apool_or_flatten(layer):
                        # It is assumed that the first layer is always
                        # conv or fc.
                        if l == 0:
                            self.state_s[l].append(layer(self.input[:, t]))
                        else:
                            self.state_s[l].append(layer(self.state_s[l - 1][-1]))

                    # Max pool
                    elif isinstance(layer, nn.MaxPool2d):
                        # It is assumed that the first layer is always
                        # conv or fc.
                        pool_result, max_index = layer(self.state_s[l - 1][-1])
                        layer.max_index_list.append(max_index)
                        self.state_s[l].append(pool_result)

                # Early stopping after every neuron spiked.
                if self.target_type == "latency":
                    self.output_s_cum += self.state_s[-1][-1]
                    if (self.output_s_cum > 0).all():
                        break

            self.term_length = t + 1

            # Reshape, permute, convert the output spike train.
            self.output = torch.stack(self.state_s[-1]).permute(1, 0, 2)

            if self.multi_model:
                self.output_each_model = self.output.reshape(
                    self.num_models,
                    int(self.output.shape[0] / self.num_models),
                    *self.output.shape[1:],
                )

            # batch x time x feature
            self.calc_num_spike()

        return self.output

    # ======= ======= =======
    # Loss
    # ======= ======= ====

    def calc_loss(self, target, calc_spike_loss=False):
        """Loss function."""

        def _mse(target_type):
            """Mean squared difference per timestep.

            Uses side effects to update the following:
            - self.diff
            - self.loss
            - self.L
            """

            with torch.no_grad():
                # Apply alpha kernel and set `self.diff` as the result
                tmp1 = self.output.float() - target.float()
                tmp2 = self.time_length + self.alpha_extend
                self.diff = self.apply_alpha_kernel(tmp1)[:, :tmp2, :]

                # L = mean square difference per timestep
                if target_type == "train":
                    self.L = torch.pow(self.diff, 2)
                elif target_type == "count":
                    self.L = torch.pow(self.diff[:, -1, :])
                else:
                    raise ValueError(
                        "expected 'train' or 'count' for MSE calculation, " +
                        f"got '{target_type}'"
                    )
                self.L /= (self.time_length * self.batch_size)

                # Summate loss
                if self.multi_model:
                    self.loss = self.L.reshape(self.num_models, -1).sum(1)
                    self.loss *= self.num_models
                else:
                    self.loss = self.L.sum()


        self.batch_size = target.shape[0]

        # Loss for spike-train-based learning rule
        if self.target_type == "train":
            _mse("train")

        # Loss for count-based learning rule
        elif self.target_type == "count":
            _mse("count")

        # Loss for latency-based learning rule
        #
        # Latency loss is defined as the cross-entropy of the softmax
        # of negatively weighted first spike timings of output neurons
        elif self.target_type == "latency":
            loss = nn.CrossEntropyLoss(reduction="none")  # lambda
            # (term_length - t_first),
            #   =term_length for the first time step,
            #   =1 for the last time step,
            #   =0 for no spike
            self.tl_m_tf = (
                (
                    self.output * torch.arange(self.term_length, 0, -1)
                    .view(1, -1, 1)
                )
                .max(dim=1)
                .values
            )

            # Softmax'd input to CE loss function
            self.sm_inp = self.tl_m_tf.float() * self.softmax_beta
            self.sm_inp.requires_grad = True

            # Compute CE loss, adjusting for batch size and model count
            self.celoss_per_batch = loss(self.sm_inp, target)
            self.celoss_per_batch /= self.batch_size
            self.celoss_per_batch *= self.num_models

            # Aggregate CE loss per model
            if self.multi_model:
                self.celoss_per_model = self.celoss_per_batch.reshape(
                    self.num_models, -1
                ).sum(1)

            # Aggregate CE loss overall
            self.celoss = self.celoss_per_batch.sum()

            # Compute CE loss gradients (wrt to graph leaves)
            self.celoss.backward()

            # ?
            with torch.no_grad():
                tmp1 = torch.gather(self.output.sum(1), 1, target.view(-1, 1))
                self.L_nospike_per_batch = (tmp1 == 0).float()
                self.L_nospike_per_batch /= self.batch_size
                self.L_nospike_per_batch *= self.num_models
                self.L_nospike_per_batch *= self.lambda_nospike

                self.loss_nospike = self.L_nospike_per_batch.sum()

                if self.multi_model:
                    reshaped_L_nospike_pb = self.L_nospike_per_batch.reshape(
                        self.num_models, -1
                    )
                    self.loss_nospike_per_model = reshaped_L_nospike_pb.sum(1)
                    self.loss = self.celoss_per_model
                    self.loss += self.loss_nospike_per_model
                else:
                    self.loss = self.celoss + self.loss_nospike

    def calc_first_stime(self):
        """Calculate the timing of the first spike.

        Returns the first spike time. As a side effect, sets
        `self.first_stime_min` and `self.first_stime_mean`, which will
        either be a list or an item depending on whether it's a multi-
        or single- model, respectively.

        Args
            self.output : [batch x time x feature]
            times : [batch]
        """
        decreasing_output = torch.arange(self.term_length, 0, -1)
        decreasing_output = decreasing_output.view(1, -1, 1) * self.output

        max_each_neuron = decreasing_output.max(dim=1).values
        # batch x feature
        max_whole_network = max_each_neuron.max(dim=1).values
        # batch
        first_stime = self.term_length - max_whole_network
        # batch
        if self.multi_model:
            first_stime_model = first_stime.reshape(self.num_models, -1)
            self.first_stime_min = first_stime_model.min(1).values.tolist()
            self.first_stime_mean = first_stime_model.mean(1).tolist()
        else:
            self.first_stime_min = first_stime.min().item()
            self.first_stime_mean = first_stime.mean().item()

        return first_stime.int()

    def calc_num_spike(self):
        """
        Add the number of total spikes except the last layer.
        Note that this value will increase with batch_size.
        """
        num_spike_total = []
        for l, layers in enumerate(self.layers):
            layer = layers[0] if self.multi_model else layers
            if not is_conv2d_or_linear(layer):
                continue
            if self.multi_model:
                num_spike_total_each_model = (
                    torch.stack(self.state_s[l])
                    .permute(1, 0, 2)
                    .reshape(
                        self.num_models,
                        int(self.output.shape[0] / self.num_models),
                        self.term_length,
                        -1,
                    )
                    .sum(1)
                    .sum(1)
                    .sum(1)
                    .int()
                )
                num_spike_total.append(num_spike_total_each_model)
            else:
                num_spike_total.append(
                    int(torch.stack(self.state_s[l]).sum().item())
                )
        if self.multi_model:
            num_spike_total = torch.stack(num_spike_total).t().tolist()
        self.num_spike_total = num_spike_total

        first_stime = self.calc_first_stime()
        num_spike_nec = []
        for l, layers in enumerate(self.layers):
            layer = layers[0] if self.multi_model else layers
            if not is_conv2d_or_linear(layer):
                continue
            num_spike_nec_each_batch = torch.tensor(
                [
                    torch.stack(self.state_s[l])
                    .int()[: (first_stime[b] + 1), b]
                    .sum()
                    .item()
                    for b in range(self.output.shape[0])
                ]
            )
            if self.multi_model:
                num_spike_nec.append(
                    num_spike_nec_each_batch.reshape(self.num_models, -1).sum(1)
                )
            else:
                num_spike_nec.append(num_spike_nec_each_batch.sum().item())
        if self.multi_model:
            num_spike_nec = torch.stack(num_spike_nec).t().tolist()
        self.num_spike_nec = num_spike_nec

    def apply_alpha_kernel(self, input, padding=True, flip=True, prime=False):
        """
        Applies self.kernel or self.kernel_prime to given input.
        Arg
            input : input.shape = [batch_size, num_time_step, num_features]
            padding : Whether apply zero padding at both sides or not.
            flip : Whether apply flip to the kernel or not.
                Usually flip is used for inference and unflipped kernel is used
                for BP.
            prime : Whether to use kernel_prime or not.
        Return
            output : result.shape = [batch_size, num_time_step+alpha, num_features]
        """
        if len(input.shape != 3):
            raise ValueError(
                f"input shape must be rank 3, was {len(input.shape)}"
            )
        batch_size, num_time_step, num_features = input.shape

        kernel = self.alpha_kernel_prime if prime else self.alpha_kernel
        kernel = kernel.flip(2) if flip else kernel
        padding_length = kernel.numel() - 1 if padding else 0

        input = input.float().permute(0, 2, 1)
        input = input.reshape(batch_size * num_features, 1, num_time_step)

        output = F.conv1d(input, kernel, padding=padding_length)
        output = output.reshape(batch_size, num_features, -1)
        output = output.permute(0, 2, 1)

        return output

    def clean_state(self):
        """Reset network."""
        self.input = None              #
        self.target = None             #
        self.output = None             #
        self.output_each_model = None  #
        self.output_s_cum = None       # cumulative spike output

        self.weight_list = None  #
        self.bias_list = None    #
        self.weight_grad = None  #
        self.bias_grad = None    #

        self.state_i = None        # I: synaptic current
        self.state_v = None        # V: membrane voltage/potential
        self.state_v_prime = None  #
        self.state_s = None        # S: spikes (binary)

        self.state_i_grad = None          #
        self.state_v_grad = None          #
        self.state_v_dep_grad = None      #
        self.state_s_grad = None          #
        self.state_t_grad = None          #
        self.state_v_grad_epr_ef1 = None  #
        self.state_v_grad_epr_ef2 = None  #
        self.dLdS = None                  # loss gradient wrt spiking
        self.dLdT = None                  # loss gradient wrt timing

        gc.collect()  # run garbage collector

    # ======= ======= =======
    # Backward Pass
    # ======= ======= ====

    def _init_backward(self):
        """Prepare network for gradient computation."""
        self.state_i_grad = list()
        self.state_v_grad = list()

        if self.lrule != "Timing":
            self.state_s_grad = list()
        else:
            self.state_s_grad = None

        if self.lrule != "RNN":
            self.state_v_dep_grad = list()
        else:
            self.state_v_dep_grad = None

        if self.lrule in ["Timing", "ANTLR"]:
            self.state_t_grad = list()
            self.state_v_grad_epr_ef1 = list()
            self.state_v_grad_epr_ef2 = list()
        else:
            self.state_t_grad = None
            self.state_v_grad_epr_ef1 = None
            self.state_v_grad_epr_ef2 = None

        for l in range(self.num_layer):
            # state_*_grad[l]: time x batch x neuron
            zeros = torch.zeros(
                self.term_length,
                self.batch_size,
                *self.fmap_shape_list[l],
                requires_grad=False,
            )

            if self.state_s_grad is not None:
                self.state_s_grad.append(zeros.clone())

            if self.state_t_grad is not None:
                self.state_t_grad.append(zeros.clone())

            if self.fmap_type_list[l] in ["conv", "fc"]:
                self.state_i_grad.append(zeros.clone())
                self.state_v_grad.append(zeros.clone())

                if self.state_v_dep_grad is not None:
                    self.state_v_dep_grad.append(zeros.clone())

                if self.state_v_grad_epr_ef1 is not None:
                    self.state_v_grad_epr_ef1.append(zeros.clone())

                if self.state_v_grad_epr_ef2 is not None:
                    self.state_v_grad_epr_ef2.append(zeros.clone())

            else:
                self.state_i_grad.append(None)
                self.state_v_grad.append(None)

                if self.state_v_dep_grad is not None:
                    self.state_v_dep_grad.append(None)

                if self.state_v_grad_epr_ef1 is not None:
                    self.state_v_grad_epr_ef1.append(None)

                if self.state_v_grad_epr_ef2 is not None:
                    self.state_v_grad_epr_ef2.append(None)

    def calc_and_set_dLdS(self):
        """Calculate the gradient of the loss wrt spiking."""

        # Case 1: latency target
        if self.target_type == "latency":
            self.dLdS = (
                torch.zeros(self.batch_size, *self.fmap_shape_list[-1])
                .scatter_(1, target.view(-1, 1), -self.L_nospike_per_batch)
                .repeat(self.term_length, 1, 1)
            )
            # time x batch x neuron
            return

            "once I failed I lost the fear of failure as a motivator" #[!]

        # Case 2: train or count target
        if self.target_type == "train":
            dLdS = self.apply_alpha_kernel(
                self.diff, padding=False, flip=False
            )
        elif self.target_type == "count":
            # bc alpha_exp = 1 here, alpha kernel application
            # is reduced to the following (I think that's what
            # the original code comment meant)
            dLdS = self.diff[:, -1, :].view(self.batch_size, 1, -1)
            dLdS = dLdS.repeat(1, self.time_length, 1)

        dLdS *= 2 / (self.time_length * self.batch_size)

        if self.multi_model:
            dLdS *= self.num_models

        # Reshape to [time, batch, neuron]
        self.dLdS = dLdS.permute(1, 0, 2)

    def calc_and_set_dLdT(self):
        """Calculate the gradient of the loss wrt timings."""

        # Case 1: latency target
        if self.target_type == "latency":
            dLdT = torch.zeros(
                self.term_length, self.batch_size, *self.fmap_shape_list[-1]
            )
            self.sm_grad = self.sm_inp.grad
            self.sm_grad[self.tl_m_tf == 0] = 0
            idxs = torch.clamp(
                self.term_length - self.tl_m_tf, 0, self.term_length - 1
            ).long()
            self.dLdT = dLdT.scatter_(
                0,
                idxs.unsqueeze(0),
                self.sm_grad.unsqueeze(0) * (-self.softmax_beta),
            )
            return  # time x batch x neuron

        # Case 2: train or count target
        if self.target_type == "train":
            dLdT_raw = self.apply_alpha_kernel(
                self.diff, flip=False, prime=True
            )
            idx1 = self.alpha_kernel_prime.numel() - 2
            idx2 = idx1 + self.time_length

            dLdT_raw = dLdT_raw[:, idx1:idx2, :,]
            dLdT_raw *= (-2) / (self.time_length * self.batch_size)
            # time x batch x neuron
        elif self.target_type == "count":
            dLdT_raw = torch.zeros(self.output.shape)

        if self.multi_model:
            dLdT_raw *= self.num_models

        dLdT = self.output * dLdT_raw
        # Reshape to [time, batch, neuron]
        self.dLdT_raw = dLdT_raw.permute(1, 0, 2)
        self.dLdT = dLdT.permute(1, 0, 2)

    def backward_custom(self, target, epoch=0):
        """
        Calculate gradient of each parameters.

        Args
            output : output value.
                output.shape = [batch_size, num_time_step, num_features]
            target : target value.
                for train and count target:
                    target.shape = [batch_size, num_time_step, num_features]
                for latency target:
                    target.shape = [batch_size]
        """

        # Initialize backward variables when batch_size is changed.
        if self.multi_model:
            if self.target_type == "latency":
                assert target.dim() == 2
                target = target.reshape(target.shape[0] * target.shape[1])
            else:
                assert target.dim() == 4
                target = target.reshape(
                    target.shape[0] * target.shape[1], *target.shape[2:]
                )

        self.batch_size = target.shape[0]
        assert self.input.shape[0] == self.batch_size

        self._init_param_grads()
        self._init_backward()

        # batch_size, num_time_step, num_features = self.output.shape
        if self.target_type in ["train", "count"]:

            # Make sure output and target shapes match
            if self.output.shape != target.shape:
                raise Exception(
                    f"output/target shape mismatch: {self.output.shape} vs " +
                    f"{target.shape}"
                )
            with torch.no_grad():
                # batch_size, num_time_step, num_features = self.output.shape
                self.calc_loss(target)
                self.calc_and_set_dLdS()
                self.calc_and_set_dLdT()

        elif self.target_type == "latency":
            if self.output.shape[0] != target.shape[0]:
                raise Exception(
                    f"output/target shape mismatch: {self.output.shape[0]} " +
                    f"vs {target.shape[0]}"
                )
            self.calc_loss(target)
            with torch.no_grad():
                self.calc_and_set_dLdS()
                self.calc_and_set_dLdT()

        # Add and clip gradients
        with torch.no_grad():

            # As per Kim et al. (2020), three learning rules classes
            # are supported for the sake of comparison: activation,
            # timing, and hybrid (ANTLR).
            if self.lrule == "Activation":
                self.gradAdd(self.dLdS, self.lrule)
            elif self.lrule == "Timing":
                self.gradAdd(self.dLdT, "Timing")
            elif self.lrule == "ANTLR":
                self.lambda_timing = 1
                self.lambda_act = 1
                self.gradAdd((self.dLdT, self.dLdS), "ANTLR")
            else:
                raise ValueError(f"invalid learning rule: '{self.lrule}'")

            # Clamp each layer's gradients to [-|grad_clip|, +|grad_clip|]
            grad_clip = self.grad_clip
            for l, layer in enumerate(self.layers):
                if self.multi_model:
                    # Ensure the (list of) gradient clip(s) can be
                    # mapped over all the provided models
                    if type(grad_clip) == list:
                        if self.num_models % len(grad_clip) != 0:
                            raise ValueError(
                                f"len(grad_clip) must be a factor of " +
                                f"self.num_models (were {len(grad_clip)} " +
                                f"and {self.num_models})"
                            )
                    # Map a gradient-clipping operation over each model
                    for m in range(self.num_models):
                        # When given a list of gradient clips, select
                        # the appropriate one for the model
                        if type(grad_clip) == list:
                            idx = m // (self.num_models // len(grad_clip))
                            grad_clip = grad_clip[idx]

                        # Clamp gradients to [-grad_clip, +grad_clip]
                        if is_conv2d_or_linear(layer):
                            clamp_grad(layer[m].weight, grad_clip)
                            clamp_grad(self.state_v_bs[l][m], grad_clip)
                else:
                    # When given a list of gradients clips for a single
                    # model, just use the first element
                    if type(grad_clip) == list:
                        grad_clip = grad_clip[0]

                    # Clamp gradients to [-grad_clip, +grad_clip]
                    if is_conv2d_or_linear(layer):
                        clamp_grad(layer.weight, grad_clip)
                        clamp_grad(self.state_v_bs[l], grad_clip)

    def gradAdd(self, output_grad_extrn, lrule, scale=1.0):
        """Update weight and bias gradients in the network."""

        def gradAdd_single_layer(layer, state_v_bs, bias_grad, weight_grad):
            """Update weight and bias gradients in a layer."""

            # Operation only applicable to Conv2d or Linear layers
            if not is_conv2d_or_linear(layer):
                return

            # Scale and apply weight gradient
            if is_nonetype(layer.weight.grad):
                # Set as gradient if none exists
                layer.weight.grad = weight_grad * scale
            else:
                # Add to existing gradient
                layer.weight.grad += weight_grad * scale

            # Scale and apply bias gradient (if specified)
            if hasattr(self, "bias_grad"):
                if is_nonetype(state_v_bs.grad):
                    # Set as gradient if none exists
                    state_v_bs.grad = bias_grad * scale
                else:
                    # Add to existing gradient
                    state_v_bs.grad += bias_grad * scale


        # output_grad_extrn: time x batch x neuron
        with torch.no_grad():

            # Run backpropagation with one of the three supported
            # learning rules (activation-based, timing-based, hybrid)
            if lrule == "Activation":
                self.bpAct(output_grad_extrn, "SRM")
            elif lrule == "Timing":
                self.bpTiming_recurrent(output_grad_extrn)
            elif lrule == "ANTLR":
                self.bpANTLR(output_grad_extrn)

            # Update every layer's weight/bias gradients
            for l, layers in enumerate(self.layers):
                if self.multi_model:
                    for m, layer in enumerate(layers):
                        gradAdd_single_layer(
                            layer,
                            self.state_v_bs[l][m],
                            self.bias_grad[l][m],
                            self.weight_grad[l][m]
                        )
                else:
                    gradAdd_single_layer(
                        layers,
                        self.state_v_bs[l],
                        self.bias_grad[l],
                        self.weight_grad[l]
                    )

    def surr_deriv(self, input_v):
        """Surrogate derivative.

        σ(V[t]) ~= dS[t] / dV[t]

        exp(-β * |V - 1|) * α
        """
        with torch.no_grad():
            output = (-self.surr_beta * (input_v - 1.0).abs()).exp()
            output *= self.surr_alpha
        return output

    def bpAct(self, output_grad_extrn, lrule):
        """Activation-based backpropagation as described in Kim et al,
        2020, section 2.2.1.
        """

        # output_grad_extrn: time x batch x neuron
        with torch.no_grad():
            for t in range(self.term_length - 1, -1, -1):
                for l in range(self.num_layer - 1, -1, -1):
                    ###### dL/dS[t] from upper layer
                    if l == self.num_layer - 1:
                        self.state_s_grad[l][t] = output_grad_extrn[t]
                    elif self.fmap_type_list[l + 1] in ["fc", "conv"]:
                        self.prop_dLdI_to_dLdX(t, l, X="S")
                    else:
                        # For pooling & flatten.
                        self.prop_dLdX_to_dLdX(t, l, X="S")

                    if self.fmap_type_list[l] in ["fc", "conv"]:
                        ###### dL/dV[t] from dL/dS[t]
                        self.state_v_grad[l][t] = (
                            self.surr_deriv(self.state_v[l][t])
                            * self.state_s_grad[l][t]
                        )
                        ###### dL/dI[t] from dL/dV[t]
                        self.prop_dLdV_to_dLdI(lrule, t, l)
            self.prop_dLdI_to_dLdW(lrule)

    def bpTiming_recurrent(self, output_grad_extrn):
        """Timing-based backpropagation as described in Kim et al,
        2020, section 2.2.2.
        """

        # output_grad_extrn: time x batch x neuron
        with torch.no_grad():
            for t in range(self.term_length - 1, -1, -1):
                for l in range(self.num_layer - 1, -1, -1):
                    ###### dL/dT[t] from upper layer
                    if l == self.num_layer - 1:
                        self.state_t_grad[l][t] = output_grad_extrn[t]

                    ###### dL/dT[t] from upper layer dL/dV[t]
                    elif self.fmap_type_list[l + 1] in ["fc", "conv"]:
                            self.tprop_dLdV_to_dLdT(t, l)
                    else:
                        self.prop_dLdX_to_dLdX(t, l, X="T")

                    if self.fmap_type_list[l] in ["fc", "conv"]:
                        ###### dL/dV[t] from dL/dT[t]
                        effective_input = self.state_s[l][t] == 1
                        self.state_v_grad[l][t] = 0
                        self.state_v_grad[l][t][effective_input] = (
                            self.state_t_grad[l][t][effective_input]
                            / -self.state_v_prime[l][t][effective_input]
                        )

                        ###### dL/dI[t] from dL/dV[t]
                        self.prop_dLdV_to_dLdI("SRM", t, l)
            self.prop_dLdI_to_dLdW("SRM", True)

    def bpANTLR(self, output_grad_extrn):  # timing + SRM (not RNN)
        """
        Perform the ANTLR computation for each layer across time.
        """
        def bpANTLR_single_layer(layer_idx, timestep):
            """
            Perform the ANTLR computation for a single layer at a
            single point in time.
            """
            l = layer_idx
            t = timestep

            ###### dL/dT[t], dL/dS[t] from upper layer
            if l == self.num_layer - 1:
                self.state_t_grad[l][t] = output_grad_extrn[0][t]
                self.state_s_grad[l][t] = output_grad_extrn[1][t]
            elif self.fmap_type_list[l + 1] in ["fc", "conv"]:
                ###### dL/dT[t] from upper layer dL/dV[t]
                self.tprop_dLdV_to_dLdT(t, l)
                ###### dL/dS[t] from upper layer dL/dV[t]
                self.prop_dLdI_to_dLdX(t, l, X="S")
            else:
                # For pooling & flatten
                self.prop_dLdX_to_dLdX(t, l, X="T")
                self.prop_dLdX_to_dLdX(t, l, X="S")

            if self.fmap_type_list[l] in ["fc", "conv"]:
                ###### dL/dV[t] from dL/dT[t], dLdS[t]
                effective_input = self.state_s[l][t] == 1

                act_vgrad = self.surr_deriv(self.state_v[l][t])
                act_vgrad *= self.state_s_grad[l][t]

                tim_vgrad = torch.zeros(self.state_t_grad[l][t].shape)
                tim_vgrad[effective_input] = (
                    self.state_t_grad[l][t][effective_input]
                    / -self.state_v_prime[l][t][effective_input]
                )

                self.state_v_grad[l][t] = self.lambda_act * act_vgrad
                self.state_v_grad[l][t][effective_input] += (
                    self.lambda_timing * tim_vgrad[effective_input]
                )

                ###### dL/dI[t] from dL/dV[t]
                self.prop_dLdV_to_dLdI("SRM", t, l)

        # output_grad_extrn: 2 (dLdT, dLdS) x time x batch x neuron
        with torch.no_grad():
            for t in range(self.term_length - 1, -1, -1):
                for l in range(self.num_layer - 1, -1, -1):
                    bpANTLR_single_layer(l, t)

            self.prop_dLdI_to_dLdW("SRM")
            ###### timing weight not used

    def prop_dLdI_to_dLdX(self, time, layer, X):
        """General function propagating gradient with respect to
        synaptic current (dLdI) to either the gradient w.r.t. binary
        spikes (S) or ??? (T)."""
        t = time
        l = layer

        if X == "S":
            state_i_grad = self.state_i_grad
            state_x_grad = self.state_s_grad
        elif X == "T":
            state_i_grad = self.state_v_grad_epr_ef2
            state_x_grad = self.state_t_grad
        else:
            raise ValueError(f"invalid X in derivative propagation: '{X}'")

        with torch.no_grad():
            if self.fmap_type_list[l + 1] == "fc":
                if self.multi_model:
                    state_i_grad_rs = state_i_grad[l + 1][t].reshape(
                        self.num_models,
                        int(self.batch_size / self.num_models),
                        -1
                    )
                    x_grad_per_beta_i = torch.bmm(
                        state_i_grad_rs, self.weight_list[l + 1]
                    ).reshape(self.batch_size, -1)
                else:
                    x_grad_per_beta_i = torch.mm(
                        state_i_grad[l + 1][t], self.layers[l + 1].weight
                    )
                state_x_grad[l][t] = self.beta_i * x_grad_per_beta_i
            elif self.fmap_type_list[l + 1] == "conv":

                self.no_mm_support("conv")  # no multi-model support

                padding = self.layers[l + 1].padding
                x_grad_per_beta_i = torch.nn.grad.conv2d_input(
                    state_x_grad[l][t].shape,
                    self.layers[l + 1].weight,
                    state_i_grad[l + 1][t],
                    padding=padding,
                )
                state_x_grad[l][t] = self.beta_i * x_grad_per_beta_i

            # t_grad of spiked timestep should only be nonzero value.
            if X == "T":
                state_x_grad[l][t] *= self.state_s[l][t]

    def tprop_dLdV_to_dLdT(self, time, layer):
        t = time
        l = layer
        with torch.no_grad():

            # If this is not the final timestep...
            if time != self.term_length - 1:  # zero-indexed

                # X <-  (1 - S[t]) * -(-0.5 * dV[t + 1])
                self.state_v_grad_epr_ef1[l + 1][t] = -(
                    -self.state_v_grad[l + 1][t + 1] / 2
                ) * (
                    1 - self.state_s[l + 1][t]
                )  # for dLdV[x+t] * -eps[t-1]

                # Y <-   α_v * (1 - S[t]) * X
                self.state_v_grad_epr_ef1[l + 1][t] += (
                    self.alpha_v
                    * (1 - self.state_s[l + 1][t])
                    * self.state_v_grad_epr_ef1[l + 1][t + 1]
                )

                # Z <-  α_i * (1 - S[t]) * Y
                self.state_v_grad_epr_ef2[l + 1][t] = (
                    self.alpha_i
                    * (1 - self.state_s[l + 1][t])
                    * self.state_v_grad_epr_ef2[l + 1][t + 1]
                )

            # If this is the final timestep...
            else:

                # X <-  0
                self.state_v_grad_epr_ef1[l + 1][t] = torch.zeros(
                    self.batch_size, *self.fmap_shape_list[l + 1]
                )

                # Z <-  0
                self.state_v_grad_epr_ef2[l + 1][t] = torch.zeros(
                    self.batch_size, *self.fmap_shape_list[l + 1]
                )

            # Changed the sign.
            self.state_v_grad_epr_ef1[l + 1][t] += self.alpha_v * (
                -self.state_v_grad[l + 1][t] / 2
            )  # for dLdV[x+t] * eps[t+1]
            self.state_v_grad_epr_ef2[l + 1][t] += (
                self.beta_v * self.alpha_i * (-self.state_v_grad[l + 1][t] / 2)
            )  # for dLdV[x+t] * eps[t+1]
            self.state_v_grad_epr_ef2[l + 1][t] += (
                self.beta_v * self.state_v_grad_epr_ef1[l + 1][t]
            )

            self.prop_dLdI_to_dLdX(t, l, X="T")

    def prop_dLdX_to_dLdX(self, time, layer, X):
        t = time
        l = layer

        if X == "S":
            state_x_grad = self.state_s_grad
        elif X == "T":
            state_x_grad = self.state_t_grad
        else:
            raise ValueError(f"invalid dX in derivative propagation: 'd{X}'")

        with torch.no_grad():
            fmap_type = self.fmap_type_list[l + 1]

            # Pooling layer
            if "pool" in fmap_type:
                kernel_size = self.layers[l + 1].kernel_size

                if fmap_type == "apool":
                    x_grad = F.interpolate(
                        state_x_grad[l + 1][t],
                        scale_factor=kernel_size
                    )
                    x_grad /= kernel_size * kernel_size

                elif fmap_type == "mpool":
                    x_grad = F.max_unpool2d(
                        state_x_grad[l + 1][t],
                        self.layers[l + 1].max_index_list[t],
                        kernel_size=kernel_size,
                        output_size=self.fmap_shape_list[l][1:],
                    )

                if list(x_grad.shape[1:]) != self.fmap_shape_list[l]:
                    to_pad = self.fmap_shape_list[l][-1] - x_grad.shape[-1]
                    x_grad = F.pad(x_grad, (0, to_pad, 0, to_pad), "constant", 0)
                    assert list(x_grad.shape[1:]) == self.fmap_shape_list[l]

                state_x_grad[l][t] = x_grad

            # Flattening layer
            elif fmap_type == "flatten":
                shape = state_x_grad[l][t].shape
                state_x_grad[l][t] = state_x_grad[l + 1][t].view(shape)

    def prop_dLdV_to_dLdI(self, style, time, layer):
        with torch.no_grad():
            t = time
            l = layer

            if style == "RNN":
                ###### dL/dV[t] from next time step (dL/dV[t+1])
                if t != self.term_length - 1:
                    x = self.alpha_v * (1 - self.state_s[l][t])
                    x *= self.state_v_grad[l][t + 1]
                    self.state_v_grad[l][t] += x

                ###### dL/dI[t] from dL/dV[t]
                self.state_i_grad[l][t] = self.beta_v * self.state_v_grad[l][t]

                ###### dL/dI[t] from next time step (dL/dI[t+1])
                if t != self.term_length - 1:
                    x = self.alpha_i * (1 - self.state_s[l][t])
                    x *= self.state_i_grad[l][t + 1]
                    self.state_i_grad[l][t] += x

            elif style == "SRM":
                ###### dL/dV[t] from next time step (dL/dV[t+1])
                self.state_v_dep_grad[l][t] = self.state_v_grad[l][t]
                if t != self.term_length - 1:
                    x = self.alpha_v * (1 - self.state_s[l][t])
                    x *= self.state_v_dep_grad[l][t + 1]
                    self.state_v_dep_grad[l][t] += x

                ###### dL/dI[t] from dL/dV[t]
                self.state_i_grad[l][t] = self.beta_v
                self.state_i_grad[l][t] *= self.state_v_dep_grad[l][t]

                ###### dL/dI[t] from next time step (dL/dI[t+1])
                if t != self.term_length - 1:
                    x = self.alpha_i * (1 - self.state_s[l][t])
                    x *= self.state_i_grad[l][t + 1]
                    self.state_i_grad[l][t] += x

            elif style == "SLAYER":
                ###### dL/dV[t] from next time step (dL/dV[t+1])
                self.state_v_dep_grad[l][t] = self.state_v_grad[l][t]
                if t != self.term_length - 1:
                    x = self.alpha_v * self.state_v_dep_grad[l][t + 1]
                    self.state_v_dep_grad[l][t] += x

                ###### dL/dI[t] from dL/dV[t]
                self.state_i_grad[l][t] = self.beta_v
                self.state_i_grad[l][t] *= self.state_v_dep_grad[l][t]

                ###### dL/dI[t] from next time step (dL/dI[t+1])
                if t != self.term_length - 1:
                    x = self.alpha_i * self.state_i_grad[l][t + 1]
                    self.state_i_grad[l][t] += x

            else:
                raise ValueError(f"invalid style: '{style}'")

    def prop_dLdI_to_dLdW(self, style, is_timing=False):
        """Propogate gradient dLdI back to dLdW."""

        with torch.no_grad():
            term_length = self.term_length

            for l in range(self.num_layer):
                if self.fmap_type_list[l] not in ["fc", "conv"]:
                    continue

                # time_length x batch x f_shape
                if l == 0:
                    hidden_s_all = self.input.transpose(0, 1)[:term_length]
                else:
                    hidden_s_all = torch.stack(self.state_s[l - 1])

                if style == "RNN":
                    v_dep_grad = self.state_v_grad[l]
                elif style == "SRM" or style == "SLAYER":
                    v_dep_grad = self.state_v_dep_grad[l]
                else:
                    raise ValueError(f"invalid style '{style}'")

                ### Calc weight grad for fc layer.
                if self.fmap_type_list[l] == "fc":
                    if self.multi_model:
                        adj_batch_size = self.batch_size // self.num_models

                        hidden_t_m_b_n = hidden_s_all.reshape(
                            term_length,
                            self.num_models,
                            adj_batch_size,
                            -1,
                        )
                        hidden_m_t_b_n = hidden_t_m_b_n.permute(1, 0, 2, 3)
                        hidden_m_tb_n = hidden_m_t_b_n.reshape(
                            self.num_models,
                            term_length * adj_batch_size,
                            -1,
                        )

                        igrad_t_m_b_n = self.state_i_grad[l].reshape(
                            term_length,
                            self.num_models,
                            adj_batch_size,
                            -1,
                        )
                        igrad_m_t_b_n = igrad_t_m_b_n.permute(1, 0, 2, 3)
                        igrad_m_tb_n = igrad_m_t_b_n.reshape(
                            self.num_models,
                            term_length * adj_batch_size,
                            -1,
                        )
                        igrad_m_n_tb = igrad_m_tb_n.permute(0, 2, 1)
                        self.weight_grad[l] = self.beta_i * torch.bmm(
                            igrad_m_n_tb, hidden_m_tb_n
                        )
                    else:
                        self.weight_grad[l] = self.beta_i * torch.mm(
                            self.state_i_grad[l]
                            .reshape(-1, self.state_i_grad[l].shape[-1])
                            .t(),
                            hidden_s_all.reshape(-1, hidden_s_all.shape[-1]),
                        )

                    if is_timing:
                        # Calculate timing penalty coefficient
                        if self.multi_model:
                            fan_in = self.weight_grad[l].shape[2]
                        else:
                            fan_in = self.weight_grad[l].shape[1]

                        timing_penalty_coeff = self.timing_penalty / fan_in

                        # ...
                        no_spike = torch.stack(self.state_s[l]).sum(dim=0) > 0
                        no_spike = 1 - no_spike.float()
                        if self.multi_model:
                            # model_batch x fan_out
                            no_spike = no_spike.reshape(
                                self.num_models,
                                int(self.batch_size / self.num_models),
                                -1,
                            )
                            no_spike_dw = (  # model x fan_out x 1
                                no_spike
                                .mean(dim=1)  # model x batch x fan_out
                                .reshape(self.num_models, -1, 1)
                            )
                        else:
                            # batch x fan_out ; fan_out x 1
                            no_spike_dw = no_spike.mean(dim=0).reshape(-1, 1)

                        no_spike_dw *= timing_penalty_coeff
                        self.weight_grad[l] -= no_spike_dw

                    if torch.isnan(self.weight_grad[l]).any():
                        self.weight_grad[l][torch.isnan(self.weight_grad[l])] = 0
                        logging.warning("nan found and replaced with 0")

                    if self.multi_model:
                        vgrad_t_m_b_n = v_dep_grad.reshape(
                            self.term_length,
                            self.num_models,
                            int(self.batch_size / self.num_models),
                            -1,
                        )
                        vgrad_m_n = vgrad_t_m_b_n.sum(0).sum(1) * self.beta_bias
                        self.bias_grad[l] = vgrad_m_n
                    else:
                        self.bias_grad[l] = (
                            v_dep_grad.sum(0).sum(0) * self.beta_bias
                        )

                ### Calc weight grad for conv layer.
                elif self.fmap_type_list[l] == "conv":

                    self.no_mm_support("conv")  # no multi-model support

                    hidden_s_all_rs = hidden_s_all.reshape(
                        -1, *list(hidden_s_all.shape[2:])
                    )
                    i_grad_rs = self.state_i_grad[l].reshape(
                        -1, *list(self.state_i_grad[l].shape[2:])
                    )
                    padding = self.layers[l].padding
                    self.weight_grad[l] = torch.nn.grad.conv2d_weight(
                        hidden_s_all_rs,
                        self.layers[l].weight.shape,
                        i_grad_rs,
                        padding=padding,
                    )
                    if torch.isnan(self.weight_grad[l]).any():
                        pdb.set_trace()
                        self.weight_grad[l][torch.isnan(self.weight_grad[l])] = 0
                        logging.warning("nan found and replaced with 0")

                    self.bias_grad[l] = (
                        v_dep_grad.sum(dim=[0, 1, 3, 4]) * self.beta_bias
                    )

    def act(self, input):
        """Activation function."""
        return (input >= 1).float()
