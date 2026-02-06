import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':

    # torch.distributed.init_process_group(backend="gloo")
    model = YOLO(
        r'D:\\qzx\\PycharmProject\\yolov5-tower\\ultralytics\\ultralytics\\cfg\\models\\v5\\yolov5n-ECBAMv2-backbone.yaml')
    # model = YOLO(
    #     r'D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\exp-tower-Coord\weights\last.pt')
    # model.load('yolov8n.pt') # loading pretrain weights
    # model.add_callback("on_train_start", freeze_layer)
    # device = torch.device('cuda:1')
    model.val(data=r'D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\rsod.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=64,
                workers=16,
                device=1,
                optimizer='auto',  # using SGD
                resume=False,  # last.pt path
                amp=True,  # close amp
                # fraction=0.2,
                project='runs/train/rsod',
                name='exp-rsod-ECBAMv2-backbone-withoutSA(n)',
                pretrained=False,
                )