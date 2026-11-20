"""Temporal 1D CNN baseline for ARIL."""

from torch import nn

from ._common import _Classifier, _EMBEDDING_SIZE, _encoder


class CSIConv1d(_Classifier):
    """Classify ARIL windows with subcarriers as channels and time as length."""

    def __init__(self, sample_shape, num_classes):
        super().__init__(sample_shape)
        self.encoder = _encoder(nn.Conv1d, sample_shape[0], nn.AdaptiveAvgPool1d(4))
        self.classifier = nn.Linear(_EMBEDDING_SIZE, num_classes)

    def forward(self, inputs):
        self._check_shape(inputs)
        return self.classifier(self.encoder(inputs))
