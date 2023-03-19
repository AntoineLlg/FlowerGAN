import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import resize


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SizeAjuster:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, image):
        return resize(image, (self.img_size, self.img_size))


class ReNormalize:  # A class diverted from the torchvision source code to reverse the normalization
    def __init__(self):
        return

    def __call__(self, tensor):
        tensor = tensor.clone()  # in order not to modify in place

        def norm_ip(img, low, high):
            img = img.clamp_(min=low, max=high)
            img = img.sub_(low).div_(max(high - low, 1e-5))
            return img

        tensor = norm_ip(tensor, float(tensor.min()), float(tensor.max()))
        return tensor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


unloader = transforms.Compose([
    ReNormalize(),
    transforms.ToPILImage()
])
