import torch
from .CRSDUN import CRSDUN

def model_generator(method, ch, n_classes, pretrained_model_path=None):
    if method == "CRSDUN":
        model = CRSDUN(stage=5, bands=ch, n_class=n_classes).cuda()

    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, weights_only=True)
        try:
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=True)
        except:
            model.load_state_dict(checkpoint, strict=True)
    return model