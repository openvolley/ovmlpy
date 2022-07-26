import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, load_classifier, time_synchronized

def get_model(weights, device='cpu', img_sz=640):
    with torch.no_grad():
        device0 = device
        device = select_device(device0)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(img_sz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16
        model.imgsz = imgsz
        model.device = device0
        return model

def detect(model, source, augment=True, conf_thres=0.25, iou_thres=0.45, agnostic_nms=True, classes=None):
    with torch.no_grad():
        device = select_device(model.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        stride = int(model.stride.max())  # model stride
        # Set Dataloader
        dataset = LoadImages(source, img_size=model.imgsz, stride=stride)
        # Get names and colors
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, model.imgsz, model.imgsz).to(device).type_as(next(model.parameters())))  # run once
        out = []
        imnum = 0
        pths = []
        for path, img, im0s, vid_cap in dataset:
            imnum += 1
            pths.append(path)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = model(img, augment=augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms, classes=classes)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (imnum, cls, *xywh, conf)
                            out.append(line)
    return [np.array(out), pths]


import cv2
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

def detect_pose(model, source, conf_thres=0.25, iou_thres=0.65):
    with torch.no_grad():
        image = cv2.imread(source)
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        device = select_device(model.device)
        image = image.to(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        image = image.half() if half else image.float()  # uint8 to fp16/32
        output, _ = model(image)
        output = non_max_suppression_kpt(output, conf_thres, iou_thres, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
    return output
