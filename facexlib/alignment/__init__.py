import cv2
import torch

from facexlib.utils import load_file_from_url
from .awing_arch import FAN
from .convert_98_to_68_landmarks import landmark_98_to_68

__all__ = ['FAN', 'landmark_98_to_68']


def init_alignment_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'awing_fan':
        model = FAN(num_modules=4, num_landmarks=98, device=device)
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['state_dict'], strict=True)
    model.eval()
    model = model.to(device)
    return model


def export_to_onnx(model_name, half=False, model_rootpath=None):
    import torch.onnx
    import onnx
    import onnxruntime
    import numpy as np

    device = 'cpu'
    model = init_alignment_model(model_name, half, device, model_rootpath)

    # img = cv2.imread('assets/test2.jpg')
    # H, W, _ = img.shape
    # offset = W / 64, H / 64, 0, 0
    #
    # img = cv2.resize(img, (256, 256))
    # inp = img[..., ::-1]
    # inp = torch.from_numpy(np.ascontiguousarray(inp.transpose((2, 0, 1)))).float()
    # inp = inp.to(device)
    # inp.div_(255.0).unsqueeze_(0)
    # torch_out = model(inp)
    #
    # print(inp.shape)

    inp = torch.randn(1, 3, 256, 256, requires_grad=True)
    torch_out = model(inp)

    torch.onnx.export(model,
                      inp,
                      "facexlib/weights/alignment_WFLW_4HG.onnx",
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output', 'boundary_channels'],
                      dynamic_axes={'input': {0: 'batch_size'}}
                      )

    onnx_model = onnx.load("facexlib/weights/alignment_WFLW_4HG.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("facexlib/weights/alignment_WFLW_4HG.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy()

    # compute ONNX Runtime output prediction
    val = inp.detach().numpy()
    print(val.shape)
    ort_inputs = {ort_session.get_inputs()[0].name: inp.detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    # We only care about 'output' and not 'boundary_channels'
    np.testing.assert_allclose(to_numpy(torch_out[0][0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[0][1]), ort_outs[1], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[0][2]), ort_outs[2], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[0][3]), ort_outs[3], rtol=1e-03, atol=1e-05)
