import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from helper.IOHelper import IOHelper
from ultralytics import YOLO


def convert_model_to_mobile():
    device = torch.device("cpu")
    model_CL01 = YOLO(r'C:\Users\USER\Downloads\CL01.pt')
    model_CL01 = model_CL01.export(format="torchscript")
    model_CL01 = torch.jit.load(r'C:\Users\USER\Downloads\CL01.torchscript')
    model_CL01 = model_CL01.to(device)
    model_CL01.eval()

    traced_script_module = torch.jit.script(model_CL01)
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter(r'C:\Users\USER\Downloads\CL01.ptl')
    print('optimize for mobile and model save completed.')


if __name__ == '__main__':
    convert_model_to_mobile()
    CLASSES = {0: "Whole-House", 1: "Roof", 2: "Wall", 3: "Door", 4: "Window", 5: "Chimney", 6: "Smoke", 7: "Fence",
               8: "Road", 9: "Pond", 10: "Mountain", 11: "Tree", 12: "Flower", 13: "Grass", 14: "Sun"}

    ioHelper = IOHelper()
    ioHelper.load_file(r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Training\Labeled\House", True)
    ioHelper.load_file(r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Validation\Labeled\House", False)
    ioHelper.create_yaml(r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Training\Original\House",
                         r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Validation\Original\House")

    model = YOLO('yolov8x.pt')
    model.train(data=r'C:\Users\USER\Desktop\2023\DeepMind\src\ML\CL01\data\CL01.yaml', epochs=100, patience=30,
                batch=1, imgsz=1280)
