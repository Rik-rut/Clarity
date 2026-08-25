import torch
from models.IFNet_HDv3 import IFNet
from models.model_pg104.GMFSS import Model as GMFSS
from models.utils.tools import *


class VFI:
    def __init__(self, model_type='rife', weights='weights', scale=1.0,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        if model_type == 'rife':
            model = IFNet()
            model.load_state_dict(convert(torch.load(f'{weights}/rife48.pkl')))
        elif model_type == 'gmfss':
            model = GMFSS()
            model.load_model(f'{weights}/train_log_pg104', -1)
        else:
            raise ValueError(f"not implement the model {model_type}")

        model.eval()
        if model_type == 'gmfss':
            model.device()
        else:
            model.to(device)

        self.model = model
        self.model_type = model_type
        base_pads = {
            'rife': 64,
            'gmfss': 128,
        }
        self.pad_size = base_pads[model_type] / scale
        self.device = device
        self.saved_result = {}
        self.scale = scale

    @torch.inference_mode()
    def gen_ts_frame(self, x, y, ts):
        _outputs = list()
        head = [x] if 0 in ts else []
        tail = [y] if 1 in ts else []
        if 0 in ts:
            ts.remove(0)
        if 1 in ts:
            ts.remove(1)
        with torch.autocast(str(self.device)):
            _reuse_things = self.model.reuse(x, y, self.scale) if self.model_type == 'gmfss' else None
            if self.model_type == 'rife':
                for t in ts:
                    scale_list = [8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]
                    _out = self.model(torch.cat((x, y), dim=1), t, scale_list)
                    _outputs.append(_out)
            elif self.model_type == 'gmfss':
                for t in ts:
                    _out = self.model.inference(x, y, _reuse_things, t)
                    _outputs.append(_out)

            _outputs = head + _outputs + tail

            return _outputs
