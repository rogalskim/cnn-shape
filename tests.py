import unittest

import torch
import torch.nn as nn

import cnn_shape


class TestCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.encoder_pool1 = nn.MaxPool2d(2, 2)
        self.encoder_conv2 = nn.Conv2d(16, 4, 3, stride=1, padding=1)
        self.encoder_pool2 = nn.MaxPool2d(2, 2)

        self.decoder_upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder_tconv1 = nn.ConvTranspose2d(4, 16, 3, stride=1, padding=1)
        self.decoder_upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder_tconv2 = nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)


class Conv2dTests(unittest.TestCase):
    def setUp(self):
        batch_size = 64
        self.input_channels = 3
        input_width = 256
        input_height = 128
        self.input = torch.zeros(size=(batch_size, self.input_channels, input_height, input_width))

    def test_output_type(self):
        layer = nn.Conv2d(self.input_channels, out_channels=2, kernel_size=3, padding=1)
        output_shape = cnn_shape.get_conv2d_output_shape(self.input, layer)
        self.assertIsInstance(output_shape, torch.Size, f"Output should be of type {torch.Size}")

    def test_shape_for_size_preserving_conv(self):
        layer = nn.Conv2d(self.input_channels, out_channels=4, kernel_size=3, padding=1)
        output_shape = cnn_shape.get_conv2d_output_shape(self.input, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_shape_for_downsampling_conv(self):
        layer = nn.Conv2d(self.input_channels, out_channels=8, kernel_size=5, padding=0)
        output_shape = cnn_shape.get_conv2d_output_shape(self.input, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_shape_for_complex_conv(self):
        layer = nn.Conv2d(self.input_channels, out_channels=4, kernel_size=3,
                          padding=(1, 0), stride=(1, 2), dilation=(1, 2))
        output_shape = cnn_shape.get_conv2d_output_shape(self.input, layer)
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
        output_shape = cnn_shape.get_maxpool2d_output_shape(self.input, layer)
        self.assertIsInstance(output_shape, torch.Size, f"Output should be of type {torch.Size}")

    def test_shape_for_basic_pool(self):
        layer = nn.MaxPool2d(kernel_size=2, stride=2)
        output_shape = cnn_shape.get_maxpool2d_output_shape(self.input, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_shape_for_dilated_pool(self):
        layer = nn.MaxPool2d(kernel_size=3, stride=2, dilation=2, padding=1)
        output_shape = cnn_shape.get_maxpool2d_output_shape(self.input, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_shape_for_complex_pool(self):
        layer = nn.MaxPool2d(kernel_size=(3, 2), stride=(3, 2), padding=(1, 0))
        output_shape = cnn_shape.get_maxpool2d_output_shape(self.input, layer)
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
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input, layer)
        self.assertIsInstance(output_shape, torch.Size, f"Output should be of type {torch.Size}")

    def test_basic_t_conv(self):
        layer = nn.ConvTranspose2d(self.input_channels, out_channels=16, kernel_size=3, padding=1)
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_dilated_t_conv(self):
        layer = nn.ConvTranspose2d(self.input_channels, out_channels=8, kernel_size=2, dilation=2)
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_striding_t_conv(self):
        layer = nn.ConvTranspose2d(self.input_channels, out_channels=8, kernel_size=2, stride=2)
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)

    def test_complex_t_conv(self):
        layer = nn.ConvTranspose2d(self.input_channels, out_channels=24, kernel_size=(2, 3), stride=(3, 2),
                                   padding=(0, 1), dilation=(3, 2), output_padding=(0, 1))
        output_shape = cnn_shape.get_conv_transpose2d_output_shape(self.input, layer)
        expected_shape = layer.forward(self.input).shape
        self.assertEqual(output_shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
