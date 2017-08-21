import pandas as pd
import cv2
import numpy as np
import pickle
from os import path
from collections import Counter

SCALE = 2.0
PATCH_SIZE = (512, 512)
ONLY_CLASSES = [10]
DISPLAY = False

DLR_IDS = {10: 'car',
           11: 'car with trailer',
           22: 'truck',
           23: 'truck with trailer',
           17: 'van with trailer',
           20: 'long truck',
           30: 'bus',
           110: 'lane lines'}


def bbox(xx, yy, ww, hh, aa):
    aa_rad = np.deg2rad(aa)
    cc, ss = np.cos(aa_rad), np.sin(aa_rad)
    res = np.array([[ww, hh], [ww, -hh], [-ww, -hh], [-ww, hh]]).dot(np.array([[cc, ss], [-ss, cc]]))
    ul, dr = np.min(res, axis=0), np.max(res, axis=0)
    return ul[0] + xx, ul[1] + yy, dr[0] + xx, dr[1] + yy


def tile_1d(length, chunk_size):
    n_tiles = (length - 1) // chunk_size + 1
    starts = np.int32(np.round(np.linspace(0, length - chunk_size, n_tiles)))
    ends = [length] if length <= chunk_size else starts + chunk_size
    for item in zip(starts, ends):
        yield item


def tile_2d(size, chunk_size):
    for tile_0 in tile_1d(size[0], chunk_size[0]):
        for tile_1 in tile_1d(size[1], chunk_size[1]):
            yield np.array(tile_0 + tile_1)[[2, 0, 3, 1]]


def create_dataset(csv_file, images_path, set_label):
    df = pd.read_csv(csv_file)

    # cleanup
    df = df.drop(df[df['size.height'] <= 0].index)
    df = df.drop(df[df['size.width'] <= 0].index)

    # classes which we are interested in
    df = df[df['class.id'].isin(ONLY_CLASSES)]

    image_files = df['file'].unique()

    image_id = 0
    records = []
    classes_count = Counter()

    for image_file in image_files:

        print('    processing %s' % image_file)

        img = cv2.imread(image_file)

        box_data = df[df['file'] == image_file]

        boxes = np.array([list(bbox(box['center.x'], box['center.y'], box['size.width'], box['size.height'], box['angle']))
                          for _, box in box_data.iterrows()])

        classes = box_data['class.id'].values

        boxes *= SCALE

        scaled_img = cv2.resize(img,
                                (int(img.shape[1] * SCALE), int(img.shape[0] * SCALE)),
                                interpolation=cv2.INTER_LANCZOS4)

        for tile in tile_2d(scaled_img.shape[:2], PATCH_SIZE):
            img_tile = np.copy(scaled_img[tile[1]:tile[3], tile[0]:tile[2], :3])
            assert(img_tile.shape[0] == PATCH_SIZE[0])
            assert(img_tile.shape[1] == PATCH_SIZE[1])
            # compute boxes intersections with tiles
            intersection_width = np.maximum(0, np.minimum(boxes[:, 2], tile[2]) - np.maximum(boxes[:, 0], tile[0]))
            intersection_height = np.maximum(0, np.minimum(boxes[:, 3], tile[3]) - np.maximum(boxes[:, 1], tile[1]))
            intersection_areas = intersection_height * intersection_width
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            intersection_fractions = intersection_areas / box_areas

            idx = intersection_fractions > 0.5

            if not np.any(idx):
                continue

            record = {}
            file_path = path.join(images_path, '%06d.jpg' % image_id)
            cv2.imwrite(file_path, img_tile)
            record['filepath'] = file_path
            record['height'] = img_tile.shape[0]
            record['width'] = img_tile.shape[1]
            record['imageset'] = set_label
            image_id += 1

            local_boxes = boxes[idx, :] - tile[[0, 1, 0, 1]]
            local_classes = classes[idx]

            boxes_list = []
            for cls, box in zip(local_classes, np.round(local_boxes).astype(np.int)):
                class_name = DLR_IDS[cls]
                classes_count[class_name] += 1

                boxes_list.append({'class': DLR_IDS[cls],
                                   'difficult': False,
                                   'x1': box[0], 'x2': box[2], 'y1': box[1], 'y2': box[3]})
            record['bboxes'] = boxes_list
            records.append(record)
            if DISPLAY:
                for box in local_boxes:
                    cv2.rectangle(img_tile,
                                  (int(round(box[0])), int(round(box[1]))),
                                  (int(round(box[2])), int(round(box[3]))),
                                  (0, 255, 0), 1)
                cv2.imshow('sample', img_tile)
                cv2.waitKey(0)
    return records, classes_count

if DISPLAY:
    cv2.namedWindow('sample', cv2.WINDOW_NORMAL)

print('Processing training data')
train_records, train_classes_count = create_dataset('dataset_train.csv', 'dlr/train', 'trainval')
print('Processing test data')
test_records, test_classes_count = create_dataset('dataset_test.csv', 'dlr/test', 'test')


records = train_records + test_records
classes_count = train_classes_count + test_classes_count
class_mapping = dict(zip(classes_count.keys(), range(len(classes_count))))

# store
with open('dlr/dlr.pkl', 'wb') as pf:
    pickle.dump({'all_imgs': records,
                 'classes_count': dict(classes_count),
                 'class_mapping': class_mapping},
                pf)
