"""2D CNN baseline for UT-HAR, NTU-Fi HAR, and WiAR."""

from torch import nn

from ._common import _Classifier, _EMBEDDING_SIZE, _encoder


class CSIConv2d(_Classifier):
    """Classify channel-first CSI windows in the adapter's native axis order."""

    def __init__(self, sample_shape, num_classes):
        super().__init__(sample_shape)
        self.encoder = _encoder(nn.Conv2d, sample_shape[0], nn.AdaptiveAvgPool2d((2, 2)))
        self.classifier = nn.Linear(_EMBEDDING_SIZE, num_classes)

    def forward(self, inputs):
        self._check_shape(inputs)
        return self.classifier(self.encoder(inputs))
