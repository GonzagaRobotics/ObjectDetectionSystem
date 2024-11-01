from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("C:\\Users\\Spencer\\Desktop\\tag2\\data\\yolo11x.pt")
    model.train(data="C:\\Users\\Spencer\\Desktop\\tag2\\data\\all\\data.yaml", epochs=100, imgsz=640)
    model.save("C:\\Users\\Spencer\\Desktop\\tag2\\data\\domo.pt")