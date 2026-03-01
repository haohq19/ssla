import torch


def load_checkpoint(model, checkpoint_path):
    # avoid loading non-parameter registered buffers
    param_names = [p[0] for p in model.named_parameters()]

    checkpoint = torch.load(checkpoint_path)
    state_dict = {k:v for k,v in checkpoint['state_dict'].items() if k in param_names}
    model.load_state_dict(state_dict, strict=False)
    return model