import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    # torch.distributed.init_process_group(backend="gloo")
    model = YOLO(
        r'D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\tower-tiny\exp-tower-raw(n)\weights\best.pt')
    # model = YOLO(
    #     r'D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\tower-tiny\exp-tower-SABR\weights\last.pt')
    # model.load('yolov8n.pt') # loading pretrain weights
    # model.add_callback("on_train_start", freeze_layer)
    # device = torch.device('cuda:1')
    model.val(data=r'D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\tower\tower_test.yaml',
              split='test',
              imgsz=640,
              batch=32,
              workers=16,
              device=1,
              project='runs/test/tower_degration',
              name='exp-tower-raw')