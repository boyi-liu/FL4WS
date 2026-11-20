"""Frame-wise 2D CNN and GRU baseline for Widar 3.0 and MM-Fi."""

from torch import nn

from ._common import _Classifier, _EMBEDDING_SIZE, _encoder


class CSISequenceClassifier(_Classifier):
    """Apply a shared frame encoder followed by a GRU over the recording."""

    def __init__(self, sample_shape, num_classes):
        super().__init__(sample_shape)
        frame_shape = self.sample_shape[1:]
        self.frame_shape = (1, *frame_shape) if len(frame_shape) == 2 else frame_shape
        self.encoder = _encoder(nn.Conv2d, self.frame_shape[0], nn.AdaptiveAvgPool2d((2, 2)))
        self.temporal = nn.GRU(_EMBEDDING_SIZE, hidden_size=64, batch_first=True)
        self.classifier = nn.Linear(self.temporal.hidden_size, num_classes)

    def forward(self, inputs):
        self._check_shape(inputs)
        batch_size, steps = inputs.shape[:2]
        frames = inputs.reshape(batch_size * steps, *self.frame_shape)
        features = self.encoder(frames).reshape(batch_size, steps, _EMBEDDING_SIZE)
        _, hidden = self.temporal(features)
        return self.classifier(hidden[-1])
