from helper.IOHelper import IOHelper
from ultralytics import YOLO

if __name__ == '__main__':
    ioHelper = IOHelper()
    ioHelper.load_file(r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Training\Labeled\Tree", True)
    ioHelper.load_file(r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Validation\Labeled\Tree", False)
    ioHelper.create_yaml(r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Training\Original\Tree", r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Validation\Original\Tree")

    model = YOLO('yolov8x.pt')
    model.train(data=r'C:\Users\USER\Desktop\Changjin\DeepMind\CL02\data\CL02.yaml', epochs=100, patience=30, batch=6, imgsz=1280)
