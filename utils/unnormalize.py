
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# TODO: is this correct?
UnNormImageNet = UnNormalize(
    mean = [0.35675976, 0.37380189, 0.3764753],
    std = [0.32064945, 0.32098866, 0.32325324]
)
