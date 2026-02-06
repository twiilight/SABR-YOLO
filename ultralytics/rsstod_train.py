import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch


if __name__ == '__main__':

    # torch.distributed.init_process_group(backend="gloo")
    model = YOLO(
        r'D:\qzx\PycharmProject\yolov5-tower\ultralytics\ultralytics\cfg\models\v5\yolov5n-ffca.yaml')
    # model = YOLO(
    #     r'D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\exp-tower-Coord\weights\last.pt')
    # model.load('yolov8n.pt') # loading pretrain weights
    # model.add_callback("on_train_start", freeze_layer)
    # device = torch.device('cuda:1')
    model.train(data=r'D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\RS_STOD\rsstod.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=8,
                device=1,
                optimizer='auto',  # using SGD
                resume=False,  # last.pt path
                amp=True,  # close amp
                # fraction=0.2,
                project='runs/train/rsstod',
                name='exp-rsstod-ffca',
                pretrained=False,
                freeze=['5.seca.sobel_h.sobel_conv',
                        '8.seca.sobel_h.sobel_conv',
                        '11.seca.sobel_h.sobel_conv',
                        '5.seca.sobel_v.sobel_conv',
                        '8.seca.sobel_v.sobel_conv',
                        '11.seca.sobel_v.sobel_conv'
                        ]
                )