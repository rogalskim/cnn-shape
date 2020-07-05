import unittest

import torch
import torch.nn as nn

import cnn_shape


class Conv2dTests(unittest.TestCase):
    def setUp(self):
        batch_size = 64
        self.input_channels = 3
        input_width = 256
        input_height = 128
        self.input = torch.zeros(size=(batch_size, self.input_channels, input_height, input_width))

    def test_output_type(self):
        layer = nn.Conv2d(self.input_channels, out_channels=2, kernel_size=3, padding=1)
        output_shape = cnn_shape.get_conv2d_output_shape(self.input.shape, layer)
        self.assertIsInstance(output_shape, torch.Size, f"Output should be of type {torch.Size}")

    def test_shape_for_size_preserving_conv(self):
        layer = nn.Conv2d(self.input_channels, out_channels=4, kernel_size=3, padding=1)
        output_shape = cnn_shape.get_conv2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_shape_for_downsampling_conv(self):
        layer = nn.Conv2d(self.input_channels, out_channels=8, kernel_size=5, padding=0)
        output_shape = cnn_shape.get_conv2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_shape_for_complex_conv(self):
        layer = nn.Conv2d(self.input_channels, out_channels=4, kernel_size=3,
                          padding=(1, 0), stride=(1, 2), dilation=(1, 2))
        output_shape = cnn_shape.get_conv2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)


class MaxPool2dTests(unittest.TestCase):
    def setUp(self):
        batch_size = 1
        self.input_channels = 16
        input_width = 64
        input_height = 48
        self.input = torch.zeros(size=(batch_size, self.input_channels, input_height, input_width))

    def test_output_type(self):
        layer = nn.MaxPool2d(kernel_size=2, padding=1)
        output_shape = cnn_shape.get_maxpool2d_output_shape(self.input.shape, layer)
        self.assertIsInstance(output_shape, torch.Size, f"Output should be of type {torch.Size}")

    def test_shape_for_basic_pool(self):
        layer = nn.MaxPool2d(kernel_size=2, stride=2)
        output_shape = cnn_shape.get_maxpool2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_shape_for_dilated_pool(self):
        layer = nn.MaxPool2d(kernel_size=3, stride=2, dilation=2, padding=1)
        output_shape = cnn_shape.get_maxpool2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_shape_for_complex_pool(self):
        layer = nn.MaxPool2d(kernel_size=(3, 2), stride=(3, 2), padding=(1, 0))
        output_shape = cnn_shape.get_maxpool2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)


class ConvTranspose2dTests(unittest.TestCase):
    def setUp(self):
        batch_size = 10
        self.input_channels = 8
        input_width = 24
        input_height = 48
        self.input = torch.zeros(size=(batch_size, self.input_channels, input_height, input_width))

    def test_output_type(self):
        layer = nn.ConvTranspose2d(self.input_channels, out_channels=16, kernel_size=3, padding=1)
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input.shape, layer)
        self.assertIsInstance(output_shape, torch.Size, f"Output should be of type {torch.Size}")

    def test_basic_t_conv(self):
        layer = nn.ConvTranspose2d(self.input_channels, out_channels=16, kernel_size=3, padding=1)
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_dilated_t_conv(self):
        layer = nn.ConvTranspose2d(self.input_channels, out_channels=8, kernel_size=2, dilation=2)
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_striding_t_conv(self):
        layer = nn.ConvTranspose2d(self.input_channels, out_channels=8, kernel_size=2, stride=2)
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_complex_t_conv(self):
        layer = nn.ConvTranspose2d(self.input_channels, out_channels=24, kernel_size=(2, 3), stride=(3, 2),
                                   padding=(0, 1), dilation=3, output_padding=(0, 1))
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)


class UpsampleTests(unittest.TestCase):
    def setUp(self):
        batch_size = 8
        self.input_channels = 12
        input_width = 64
        input_height = 56
        self.input = torch.zeros(size=(batch_size, self.input_channels, input_height, input_width))

    def test_output_type(self):
        layer = nn.Upsample(scale_factor=2)
        output_shape = cnn_shape.get_upsample_output_shape(self.input.shape, layer)
        self.assertIsInstance(output_shape, torch.Size, f"Output should be of type {torch.Size}")

    def test_bilinear_upsample_shape(self):
        layer = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        output_shape = cnn_shape.get_upsample_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_nn_upsample_shape(self):
        layer = nn.Upsample(scale_factor=(2, 1.5), mode="nearest")
        output_shape = cnn_shape.get_upsample_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_bicubic_upsample_shape(self):
        layer = nn.Upsample(scale_factor=3, mode="bicubic", align_corners=True)
        output_shape = cnn_shape.get_upsample_output_shape(self.input.shape, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)


class TestCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.encoder_pool1 = nn.MaxPool2d(2, 2)
        self.decoder_upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder_tconv1 = nn.ConvTranspose2d(4, 16, 3, stride=1, padding=1)


class NetworkAnalysisTests(unittest.TestCase):
    def setUp(self):
        batch_size = 8
        self.input_channels = 12
        input_width = 64
        input_height = 56
        self.input = torch.zeros(size=(batch_size, self.input_channels, input_height, input_width))
        self.network = TestCnn()

    def test_return_type(self):
        shape_dict = cnn_shape.get_layer_output_shapes(self.input.shape, self.network)
        self.assertIsInstance(shape_dict, dict)

    def test_result_dict_has_len_equal_to_layer_count(self):
        layer_count = len(list(self.network.named_modules())[1:])
        shape_dict = cnn_shape.get_layer_output_shapes(self.input.shape, self.network)
        self.assertEqual(len(shape_dict), layer_count)

    def test_throws_when_unsupported_layer_type_encountered(self):
        self.network.add_module("unsupported", nn.AdaptiveAvgPool2d(16))
        with self.assertRaises(KeyError):
            cnn_shape.get_layer_output_shapes(self.input.shape, self.network)

    def test_ignores_modules_not_being_layers(self):
        self.network.add_module("dropout", nn.Dropout())
        cnn_shape.get_layer_output_shapes(self.input.shape, self.network)

    def test_returns_correct_shapes_for_all_layers(self):
        shape_dict = cnn_shape.get_layer_output_shapes(self.input.shape, self.network)
        expected_shape = cnn_shape.get_conv2d_output_shape(self.input.shape, self.network.encoder_conv1)
        self.assertEqual(shape_dict["encoder_conv1"], expected_shape)
        expected_shape = cnn_shape.get_maxpool2d_output_shape(expected_shape, self.network.encoder_pool1)
        self.assertEqual(shape_dict["encoder_pool1"], expected_shape)
        expected_shape = cnn_shape.get_upsample_output_shape(expected_shape, self.network.decoder_upsample1)
        self.assertEqual(shape_dict["decoder_upsample1"], expected_shape)
        expected_shape = cnn_shape.get_conv_transpose2d_output_shape(expected_shape, self.network.decoder_tconv1)
        self.assertEqual(shape_dict["decoder_tconv1"], expected_shape)


class ModuleListTests(unittest.TestCase):
    def setUp(self) -> None:
        batch_size = 3
        self.input_channels = 3
        input_width = 32
        input_height = 32
        self.input = torch.zeros(size=(batch_size, self.input_channels, input_height, input_width))

        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        self.module_list.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.module_list.append(nn.MaxPool2d(2, 2))

    def test_return_type(self):
        shape_dict = cnn_shape.get_layer_output_shapes(self.input.shape, self.module_list)
        self.assertIsInstance(shape_dict, dict)

    def test_result_dict_has_len_equal_to_layer_count(self):
        layer_count = len(self.module_list)
        shape_dict = cnn_shape.get_layer_output_shapes(self.input.shape, self.module_list)
        self.assertEqual(len(shape_dict), layer_count)

    def test_throws_when_unsupported_layer_type_encountered(self):
        self.module_list.append(nn.AdaptiveAvgPool2d(16))
        with self.assertRaises(KeyError):
            cnn_shape.get_layer_output_shapes(self.input.shape, self.module_list)

    def test_ignores_modules_not_being_layers(self):
        self.module_list.append(nn.Dropout())
        cnn_shape.get_layer_output_shapes(self.input.shape, self.module_list)

    def test_returns_correct_shapes_for_all_layers(self):
        shape_dict = cnn_shape.get_layer_output_shapes(self.input.shape, self.module_list)
        expected_shape = cnn_shape.get_conv2d_output_shape(self.input.shape, self.module_list[0])
        self.assertEqual(shape_dict["0"], expected_shape)
        expected_shape = cnn_shape.get_conv2d_output_shape(self.input.shape, self.module_list[1])
        self.assertEqual(shape_dict["1"], expected_shape)
        expected_shape = cnn_shape.get_maxpool2d_output_shape(expected_shape, self.module_list[2])
        self.assertEqual(shape_dict["2"], expected_shape)

    def test_nested_list(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.list = nn.ModuleList()
                self.list.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
                self.list.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
                self.list.append(nn.MaxPool2d(2, 2))
                self.pool = nn.MaxPool2d(2, 2)

        self.network = Net()
        shape_dict = cnn_shape.get_layer_output_shapes(self.input.shape, self.network)
        expected_shape = cnn_shape.get_conv2d_output_shape(self.input.shape, self.network.list[0])
        self.assertEqual(shape_dict["list"]["0"], expected_shape)
        expected_shape = cnn_shape.get_conv2d_output_shape(self.input.shape, self.network.list[1])
        self.assertEqual(shape_dict["list"]["1"], expected_shape)
        expected_shape = cnn_shape.get_maxpool2d_output_shape(expected_shape, self.network.list[2])
        self.assertEqual(shape_dict["list"]["2"], expected_shape)
        expected_shape = cnn_shape.get_maxpool2d_output_shape(expected_shape, self.network.pool)
        self.assertEqual(shape_dict["pool"], expected_shape)


if __name__ == '__main__':
    unittest.main()
