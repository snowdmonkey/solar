import torch
import cv2
import pkg_resources
import numpy as np
from pathlib import Path
from fcn.nets import fc_dense_net57
from torch.autograd import Variable
from collections import OrderedDict

image_folder = Path(r"C:\Users\h232559\Documents\projects\uav\pic\qingyun\dingzhoueast\2018-01-19\ir\rotated")
output_folder = Path(r"C:\Users\h232559\Desktop\test")


def main():

    files = image_folder.glob("*.jpg")
    model = fc_dense_net57(n_classes=2, channels=1)

    weight_path = r"Z:\label\model-20180206-2310-scale.pt"
    # model = torch.load(weight_path, map_location=lambda storage, loc: storage)
    state = torch.load(weight_path, map_location=lambda storage, loc: storage)

    new_state = OrderedDict()
    for k, v in state.items():
        name = k[7:]
        new_state[name] = v

    model.load_state_dict(new_state)
    # model.load_state_dict(state)
    model.eval()

    for file in files:
        raw = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        img = (raw - 127.5) / 127.5

        img = img.reshape(1, 1, img.shape[0], img.shape[1])

        tensor = torch.from_numpy(img)
        data = Variable(tensor, volatile=True).float()
        output = model(data)
        bs, c, h, w = output.size()
        _, indices = output.data.max(1)
        indices = indices.view(bs, h, w)
        output = indices.numpy()[0]

        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        overlay = raw.copy()
        overlay[output == 1] = (0, 255, 0)

        raw = cv2.addWeighted(raw, 0.7, overlay, 0.3, 0)

        cv2.imwrite(str(output_folder / file.name), raw)


if __name__ == "__main__":
    main()
