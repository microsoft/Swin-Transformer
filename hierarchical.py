import einops
import torch


def accuracy(output, target, topk=(1,), hierarchy_level=-1):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Copied from rwightman/pytorch-image-models/timm/utils/metrics.py and modified
    to work with hierarchical outputs as well.

    When the output is hierarchical, only returns the accuracy for `hierarchy_level`
    (default -1, which is the fine-grained level).
    """
    output_levels = 1
    if isinstance(output, list):
        output_levels = len(output)
        output = output[-1]
        print (output_levels)
    # print (output)

    batch_size = output.size(0)

    # Target might have multiple levels because of the hierarchy
    if target.squeeze().ndim == 2:
        assert target.squeeze().shape == (batch_size, output_levels)
        target = target[:, -1]

    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size
        for k in topk
    ]


class FineGrainedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    A cross-entropy used with hierarchical inputs and targets and only
    looks at the finest-grained tier (the last level).
    """

    def forward(self, inputs, targets):
        fine_grained_inputs = inputs[-1]
        fine_grained_targets = targets[:, -1]
        return super().forward(fine_grained_inputs, fine_grained_targets)


class HierarchicalCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, coeffs=(1.0,), **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.clone().detach().type(torch.float)
        else:
            coeffs = torch.tensor(coeffs, dtype=torch.float)

        self.register_buffer("coeffs", coeffs)

    def forward(self, inputs, targets):
        if not isinstance(targets, list):
            targets = einops.rearrange(targets, "batch tiers -> tiers batch")

        assert (
            len(inputs) == len(targets) == len(self.coeffs)
        ), f"{len(inputs)} != {len(targets)} != {len(self.coeffs)}"

        losses = torch.stack(
            [
                # Need to specify arguments to super() because of some a bug
                # with super() in list comprehensions/generators (unclear)
                super(HierarchicalCrossEntropyLoss, self).forward(input, target)
                for input, target in zip(inputs, targets)
            ]
        )

        return torch.dot(self.coeffs, losses)
