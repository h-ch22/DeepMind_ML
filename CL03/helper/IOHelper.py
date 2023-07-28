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
                id = id.replace("남자사람", "Person_M")
                id = id.replace("여자사람", "Person_F")
                id = id.replace("남", "M")
                id = id.replace("여", "F")

                bbox_count = int(data["annotations"]["bbox_count"])
                bbox = data["annotations"]["bbox"]
                gender = data["annotations"]["class"]
                bboxes = ""

                for box in range(0, bbox_count):
                    x = bbox[box]["x"]
                    y = bbox[box]["y"]
                    w = bbox[box]["w"]
                    h = bbox[box]["h"]
                    label = ""

                    if gender == "남자사람" and bbox[box]["label"] == "사람전체":
                        label = "남자 사람"

                    elif gender == "여자사람" and bbox[box]["label"] == "사람전체":
                        label = "여자 사람"

                    center_x_norm = (x + (w/2))/1280
                    center_y_norm = (y + (h/2))/1280
                    width_norm = w/1280
                    height_norm = h/1280

                    bboxes += "%s %s %s %s %s\n" % (self.__class_to_code__(bbox[box]["label"] if label == "" else label), round(center_x_norm, 7), round(center_y_norm, 7), round(width_norm, 7), round(height_norm, 7))

                label_dir = r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Training\Original\Person\labels" + r'\%s' % id + '.txt' if isTrain else r"C:\Users\USER\Desktop\Changjin\DeepMind\DATA\Validation\Original\Person\labels" + r'\%s' % id + '.txt'

                if not os.path.exists(label_dir):
                    try:
                        with open(label_dir, 'w') as f:
                            f.write(bboxes)

                    except FileNotFoundError:
                        raise Exception("File not found : %s" % label_dir)
        else:
            raise Exception("Directory not found : %s" % PATH)

    def __class_to_code__(self, class_string):
        CLASSES = {"남자 사람": 0, "여자 사람": 1, "머리": 2, "얼굴": 3, "눈": 4, "코": 5, "입": 6, "귀": 7, "머리카락": 8, "목": 9,
                   "상체": 10, "팔": 11, "손": 12, "다리": 13, "발": 14, "단추": 15, "주머니": 16, "운동화": 17,
                   "남자구두": 18, "여자구두": 19}

        return CLASSES[class_string]

    def __get_YOLO_bounding_box__(self, x, y, w, h):
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        return x1, y1, x2, y2

    def create_yaml(self, TRAIN_PATH, VALID_PATH):
        data = {'train': TRAIN_PATH,
                'val': VALID_PATH,
                'names': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'nc': 20}

        with open(r'C:\Users\USER\Desktop\Changjin\DeepMind\CL03\data\CL03.yaml', 'w') as f:
            yaml.dump(data, f)
