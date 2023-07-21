import ultralytics

from helper.IOHelper import IOHelper
from ultralytics import YOLO

if __name__ == '__main__':
    CLASSES = {0: "Whole-House", 1: "Roof", 2: "Wall", 3: "Door", 4: "Window", 5: "Chimney", 6: "Smoke", 7: "Fence",
               8: "Road", 9: "Pond", 10: "Mountain", 11: "Tree", 12: "Flower", 13: "Grass", 14: "Sun"}

    ioHelper = IOHelper()
    ioHelper.load_file(r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Training\Labeled\House", True)
    ioHelper.load_file(r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Validation\Labeled\House", False)
    ioHelper.create_yaml(r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Training\Original\House", r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Validation\Original\House")

    model = YOLO('yolov8x.pt')
    model.train(data=r'C:\Users\USER\Desktop\2023\DeepMind\src\ML\CL01\data\CL01.yaml', epochs=100, patience=30, batch=1, imgsz=1280)
