from helper.IOHelper import IOHelper
from ultralytics import YOLO

if __name__ == '__main__':
    ioHelper = IOHelper()
    ioHelper.load_file(r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Training\Labeled\Person", True)
    ioHelper.load_file(r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Validation\Labeled\Person", False)
    ioHelper.create_yaml(r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Training\Original\Person", r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Validation\Original\Person")

    model = YOLO('yolov8x.pt')
    model.train(data=r'C:\Users\USER\Desktop\Changjin\DeepMind\CL03\data\CL03.yaml', epochs=100, patience=30, batch=8, imgsz=1280)
