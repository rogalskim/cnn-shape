import typing

import torch
import torch.nn as nn


def _calculate_2d_conv_output_dim_size(dim: int, input_size: (int, int), padding: (int, int), dilation: (int, int),
                                       kernel: (int, int), stride: (int, int)) -> int:

    return (input_size[dim] + 2*padding[dim] - dilation[dim]*(kernel[dim]-1) - 1) / stride[dim] + 1


def _calculate_2d_conv_output_shape(batch_size: int, output_channels: int, input_size: (int, int), padding: (int, int),
                                    dilation: (int, int), kernel: (int, int), stride: (int, int)) -> torch.Size:

    output_height = _calculate_2d_conv_output_dim_size(0, input_size, padding, dilation, kernel, stride)
    output_width = _calculate_2d_conv_output_dim_size(1, input_size, padding, dilation, kernel, stride)
    return torch.Size((batch_size, output_channels, int(output_height), int(output_width)))


def __make_tuple(input: typing.Union[int, tuple]) -> tuple:
    if type(input) is int:
        return (input, input)
    return input


def get_conv2d_output_shape(layer_input: torch.Tensor, layer: nn.Conv2d) -> torch.Size:
    assert len(layer_input.shape) == 4, f"Input Tensor must have 4 dimensions, but has {len(layer_input)}"
    return _calculate_2d_conv_output_shape(batch_size=layer_input.shape[0], output_channels=layer.out_channels,
                                           input_size=(layer_input.shape[2], layer_input.shape[3]),
                                           padding=layer.padding, dilation=layer.dilation,
                                           kernel=layer.kernel_size, stride=layer.stride)


def get_maxpool2d_output_shape(layer_input: torch.Tensor, layer: nn.MaxPool2d) -> torch.Size:
    assert len(layer_input.shape) == 4, f"Input Tensor must have 4 dimensions, but has {len(layer_input)}"
    padding = __make_tuple(layer.padding)
    dilation = __make_tuple(layer.dilation)
    kernel = __make_tuple(layer.kernel_size)
    stride = __make_tuple(layer.stride)
    return _calculate_2d_conv_output_shape(batch_size=layer_input.shape[0], output_channels=layer_input.shape[1],
                                           input_size=(layer_input.shape[2], layer_input.shape[3]),
                                           padding=padding, dilation=dilation, kernel=kernel, stride=stride)
