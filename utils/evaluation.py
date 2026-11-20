"""Batch-wise classification metrics for local or global PyTorch models."""

from itertools import chain
from math import isfinite

import torch
from torch.nn.functional import cross_entropy


class ClassificationEvaluator:
    """Accumulate loss and a confusion matrix without retaining predictions."""

    def __init__(self, num_classes):
        if type(num_classes) is not int or num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.loss_sum = 0.0

    @torch.no_grad()
    def update(self, logits, targets):
        """Add raw logits and a vector of zero-based int64 class labels."""
        if logits.shape != (targets.numel(), self.num_classes) or targets.ndim != 1 or targets.dtype != torch.int64:
            raise ValueError("Expected logits shaped (batch, num_classes) and int64 targets shaped (batch,)")
        if not targets.numel():
            return
        if ((targets < 0) | (targets >= self.num_classes)).any().item():
            raise ValueError("Evaluation targets must be in [0, num_classes)")

        loss = cross_entropy(logits, targets, reduction="sum").item()
        if not isfinite(loss):
            raise ValueError("Evaluation produced a non-finite loss")
        pairs = targets * self.num_classes + logits.argmax(dim=1)
        self.confusion_matrix += torch.bincount(
            pairs.cpu(), minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        self.loss_sum += loss

    def compute(self):
        """Return JSON-ready metrics; rows are true labels and columns predictions."""
        matrix = self.confusion_matrix
        num_samples = matrix.sum().item()
        if not num_samples:
            raise ValueError("Evaluation requires at least one test sample")
        true_positives = matrix.diag().to(torch.float64)
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        return {
            "num_samples": num_samples,
            "loss": self.loss_sum / num_samples,
            "accuracy": true_positives.sum().item() / num_samples,
            "macro_precision": (true_positives / predicted.clamp_min(1)).mean().item(),
            "macro_recall": (true_positives / support.clamp_min(1)).mean().item(),
            "macro_f1": (2 * true_positives / (support + predicted).clamp_min(1)).mean().item(),
            "confusion_matrix": matrix.tolist(),
        }


@torch.no_grad()
def evaluate(model, batches, *, num_classes):
    """Evaluate a model on an iterable of (features, targets) batches."""
    evaluator = ClassificationEvaluator(num_classes)
    device = next(chain(model.parameters(), model.buffers()), torch.empty(0)).device
    was_training = model.training
    model.eval()
    try:
        for features, targets in batches:
            evaluator.update(model(features.to(device)), targets.to(device))
        return evaluator.compute()
    finally:
        model.train(was_training)
