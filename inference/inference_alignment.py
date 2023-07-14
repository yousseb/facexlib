import argparse
import cv2
import numpy as np
import onnxruntime
import torch

from facexlib.alignment import init_alignment_model, landmark_98_to_68, export_to_onnx
from facexlib.alignment.awing_arch import calculate_points
from facexlib.visualization import visualize_alignment


def main(args):
    if args.to_onnx:
        export_to_onnx(args.model_name)
        exit(0)

    if args.onnx:
        img = cv2.imread(args.img_path)
        H, W, _ = img.shape
        offset = W / 64, H / 64, 0, 0

        img = cv2.resize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32')
        img = img / 255.0

        ort_session = onnxruntime.InferenceSession("facexlib/weights/alignment_WFLW_4HG.onnx")
        ort_inputs = {ort_session.get_inputs()[0].name: img}
        ort_outs = ort_session.run(None, ort_inputs)
        outputs = ort_outs[:4]
        out = outputs[-1][:, :-1, :, :]

        heatmaps = out
        pred = calculate_points(heatmaps).reshape(-1, 2)
        pred *= offset[:2]
        pred += offset[-2:]
        landmarks = pred
        if args.to68:
            landmarks = landmark_98_to_68(landmarks)
        visualize_alignment(img, [landmarks], args.save_path)
        exit(0)


    # initialize model
    align_net = init_alignment_model(args.model_name, device=args.device)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        landmarks = align_net.get_landmarks(img)
        if args.to68:
            landmarks = landmark_98_to_68(landmarks)
        visualize_alignment(img, [landmarks], args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument('--save_path', type=str, default='test_alignment.png')
    parser.add_argument('--model_name', type=str, default='awing_fan')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--to68', action='store_true')
    parser.add_argument('--to-onnx', action='store_true')
    parser.add_argument('--onnx', action='store_true')

    args = parser.parse_args()

    main(args)
