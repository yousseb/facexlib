import os

import cv2
import torch

from facexlib.utils import load_file_from_url
from .bisenet import BiSeNet
from .parsenet import ParseNet
from .. import img2tensor


def init_parsing_model(model_name='bisenet', half=False, device='cuda', model_rootpath=None):
    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth'
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model


def export_to_onnx(model_name, half=False, model_rootpath=None):
    import torch.onnx
    import onnx
    import onnxruntime
    import numpy as np

    device = 'cpu'
    model = init_parsing_model(model_name, half, device, model_rootpath)
    model.eval()
    from torchvision.transforms.functional import normalize
    img_path = 'assets/test2.jpg'
    img_name = os.path.basename(img_path)
    img_basename = os.path.splitext(img_name)[0]

    img_input = cv2.imread(img_path)
    img_input = cv2.resize(img_input, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = img2tensor(img_input.astype('float32') / 255., bgr2rgb=True, float32=True)
    normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
    img = torch.unsqueeze(img, 0).cpu()

    # torch_out = model(inp)
    #
    print(img.shape)

    inp = torch.randn(1, 3, 512, 512, requires_grad=True)
    torch_out = model(inp)

    torch.onnx.export(model,
                      inp,
                      f"facexlib/weights/parsing_{model_name}.onnx",
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['out_mask', 'out_img'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'out_mask': [1],
                                    'out_img': [1]}
                      )

    onnx_model = onnx.load(f"facexlib/weights/parsing_{model_name}.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(f"facexlib/weights/parsing_{model_name}.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    # compute ONNX Runtime output prediction
    val = inp.detach().numpy()
    print(val.shape)
    ort_inputs = {ort_session.get_inputs()[0].name: inp.detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    # We only care about 'output' and not 'boundary_channels'
    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(torch_out[0][2]), ort_outs[2], rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(torch_out[0][3]), ort_outs[3], rtol=1e-03, atol=1e-05)