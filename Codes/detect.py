# # detect.py
# import time
# from pathlib import Path
# import cv2
# import torch
# from numpy import random
# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device, time_synchronized

# def detect(
#     source='0',                 # '0' for live, or file path
#     weights='D:/Real-Time-Traffic-Sign-Detection-main/Model/Model/weights/best.pt',
#     img_size=640,
#     conf_thres=0.5,
#     iou_thres=0.45,
#     device='',                  # 'cpu' or '0'
#     view_img=False,
# ):
#     """
#     Generator function that yields processed frames and detected signs.
#     """
#     webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))

#     # Initialize
#     set_logging()
#     device = select_device(device)
#     half = device.type != 'cpu'

#     # Load model
#     model = attempt_load(weights, map_location=device)
#     imgsz = check_img_size(img_size, s=model.stride.max())
#     if half:
#         model.half()

#     # Load dataset
#     if webcam:
#         dataset = LoadStreams(source, img_size=imgsz)
#     else:
#         dataset = LoadImages(source, img_size=imgsz)

#     names = model.module.names if hasattr(model, 'module') else model.names
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

#     # Warmup
#     img = torch.zeros((1, 3, imgsz, imgsz), device=device)
#     _ = model(img.half() if half else img) if device.type != 'cpu' else None

#     # For tracking which signs have been announced in current frame
#     for path, img, im0s, vid_cap in dataset:
#         img_t = torch.from_numpy(img).to(device)
#         img_t = img_t.half() if half else img_t.float()
#         img_t /= 255.0
#         if img_t.ndimension() == 3:
#             img_t = img_t.unsqueeze(0)

#         t1 = time_synchronized()
#         pred = model(img_t, augment=False)[0]
#         pred = non_max_suppression(pred, conf_thres, iou_thres)
#         t2 = time_synchronized()

#         new_signs = []

#         for i, det in enumerate(pred):
#             if webcam:
#                 im0 = im0s[i].copy()
#             else:
#                 im0 = im0s

#             if len(det):
#                 det[:, :4] = scale_coords(img_t.shape[2:], det[:, :4], im0.shape).round()

#                 for *xyxy, conf, cls in reversed(det):
#                     label = f'{names[int(cls)]} {conf:.2f}'
#                     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
#                     new_signs.append(names[int(cls)])

#             # Show FPS
#             im0 = cv2.putText(im0, f"FPS: {1/(t2-t1):.2f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#         yield im0, new_signs


# detect.py
import time
from pathlib import Path
import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def detect(
    source='0',                 # '0' for live, or file path
    weights='D:/Real-Time-Traffic-Sign-Detection-main/Model/Model/weights/best.pt',
    img_size=640,
    conf_thres=0.5,
    iou_thres=0.45,
    device='',                  # 'cpu' or '0'
    view_img=False,
    classes=None,
    agnostic_nms=False,
    augment=False
):
    """
    source: image path, video path, or '0' for webcam
    Returns a generator yielding (frame, list_of_detected_signs)
    """
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(img_size, s=model.stride.max())
    if half:
        model.half()

    # Dataloader
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Warmup
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    # Inference loop
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        current_signs = []

        for i, det in enumerate(pred):
            if webcam:
                im0 = im0s[i].copy()
            else:
                im0 = im0s

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                current_signs = [names[int(cls)] for *xyxy, conf, cls in reversed(det)]

                # Draw boxes
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Optionally put FPS
            fps = 1 / (t2 - t1)
            im0 = cv2.putText(im0, f"FPS: {fps:.2f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Yield frame and current signs
            yield im0, current_signs
