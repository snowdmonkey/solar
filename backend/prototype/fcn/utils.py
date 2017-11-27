from torch.nn import Module
from .nets import fc_dense_net57
import torch
import os


def get_panel_group_model() -> Module:

    model = fc_dense_net57(n_classes=2, channels=1)
    package_dir, _ = os.path.split(__file__)
    state = torch.load(os.path.join(package_dir, "weights", "Panel_Rest_Model_e252_cpu.pth"))
    model.load_state_dict(state["state_dict"])
    return model
