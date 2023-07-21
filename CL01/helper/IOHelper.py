import glob
import json
import os
import yaml


class IOHelper:
    def load_file(self, PATH, isTrain):
        if os.path.exists(PATH):
            LABELED_FILES = glob.glob(PATH + r'\*.json')

            for file in LABELED_FILES:
                jsonFile = open(file, 'rt', encoding='UTF8')
                data = json.load(jsonFile)

                id = data["meta"]["label_path"].split('./')[1].split('.json')[0]
                id = id.replace("집", "House")
                id = id.replace("남", "M")
                id = id.replace("여", "F")

                bbox_count = int(data["annotations"]["bbox_count"])
                bbox = data["annotations"]["bbox"]
                bboxes = ""

                for box in range(0, bbox_count):
                    x = bbox[box]["x"]
                    y = bbox[box]["y"]
                    w = bbox[box]["w"]
                    h = bbox[box]["h"]

                    center_x_norm = x/1280
                    center_y_norm = y/1280
                    width_norm = w/1280
                    height_norm = h/1280

                    bboxes += "%s %s %s %s %s\n" % (self.__class_to_code__(bbox[box]["label"]), round(center_x_norm, 7), round(center_y_norm, 7), round(width_norm, 7), round(height_norm, 7))

                label_dir = r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Training\Original\House\labels" + r'\%s' % id + '.txt' if isTrain else r"C:\Users\USER\Desktop\2023\DeepMind\DATA\Validation\Original\House\labels" + r'\%s' % id + '.txt'

                if not os.path.exists(label_dir):
                    try:
                        with open(label_dir, 'w') as f:
                            f.write(bboxes)

                    except FileNotFoundError:
                        raise Exception("File not found : %s" % label_dir)
        else:
            raise Exception("Directory not found : %s" % PATH)

    def __class_to_code__(self, class_string):
        CLASSES = {"집전체": 0, "지붕": 1, "집벽": 2, "문": 3, "창문": 4, "굴뚝": 5, "연기": 6, "울타리": 7,
                   "길": 8, "연못": 9, "산": 10, "나무": 11, "꽃": 12, "잔디": 13, "태양": 14}

        return CLASSES[class_string]

    def __get_YOLO_bounding_box__(self, x, y, w, h):
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        return x1, y1, x2, y2

    def create_yaml(self, TRAIN_PATH, VALID_PATH):
        data = {'train': TRAIN_PATH,
                'val': VALID_PATH,
                'names': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                'nc': 15}

        with open(r'C:\Users\USER\Desktop\2023\DeepMind\src\ML\CL01\data\CL01.yaml', 'w') as f:
            yaml.dump(data, f)
