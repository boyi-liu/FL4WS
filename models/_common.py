"""Shared convolutional encoder and input validation for classifiers."""

from torch import nn

_WIDTHS = (16, 32, 48)
_EMBEDDING_SIZE = 4 * _WIDTHS[-1]


def _encoder(convolution, channels, pooling):
    layers = []
    for width in _WIDTHS:
        layers.extend((
            convolution(channels, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, width),
            nn.ReLU(),
        ))
        channels = width
    return nn.Sequential(*layers, pooling, nn.Flatten(start_dim=1))


class _Classifier(nn.Module):
    def __init__(self, sample_shape):
        super().__init__()
        self.sample_shape = tuple(sample_shape)

    def _check_shape(self, inputs):
        if inputs.ndim != len(self.sample_shape) + 1 or tuple(inputs.shape[1:]) != self.sample_shape:
            raise ValueError(f"Expected batch of samples shaped {self.sample_shape}, got {tuple(inputs.shape)}")
