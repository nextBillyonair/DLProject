from torch.nn import NLLLoss


class MaskedNLLLoss(NLLLoss):
    # TODO: custom reduction
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, reduction='none')

    def forward(self, input, target, mask):
        masked = (super().forward(input, target) * mask.float()).sum()
        num_unmasked = mask.nonzero().size(0)

        # Mean of losses that weren't masked away
        return masked / num_unmasked if num_unmasked > 0 else masked
