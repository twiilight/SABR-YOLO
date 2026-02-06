from ultralytics import YOLO


def main():
    model = YOLO(
        r"D:\qzx\PycharmProject\yolov5-tower\ultralytics\runs\train\tower-tiny\exp-tower-yolov3(n)\weights\best.pt")
    # source = r"D:\qzx\PycharmProject\yolov5-tower\cfg\dataset\tower_noise_2"
    source = r"D:\qzx\tower_dataset\tower\images\138762_6.jpg"
    results = model(source, visualize=False, imgsz=640, iou=0.3)  # 要预测图片路径和使用可视化
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show(labels=False)  # display to screen
        result.save(labels=True, filename='results_138762_6_v3.jpg')  # save to disk
        # result.save(labels=True)


if __name__ == '__main__':
    main()
