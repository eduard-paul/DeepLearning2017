import argparse
import pathlib
import os
import errno
import json
import re
from random import shuffle

import cv2
import numpy as np
#import caffe


def check_is_file(entry: str) -> pathlib.Path:
    entry = pathlib.Path(entry).absolute()
    if not entry.is_file():
        raise argparse.ArgumentTypeError('{}: {}'.format(os.strerror(errno.ENOENT), entry.as_posix()))

    return entry


def check_is_dir(entry: str) -> pathlib.Path:
    entry = pathlib.Path(entry).absolute()
    if not entry.is_dir():
        raise argparse.ArgumentTypeError('{}: {}'.format(os.strerror(errno.ENOTDIR), entry.as_posix()))

    return entry


def check_range(entry: str) -> float:

    entry = float(entry)
    low, high = 0.0, 1.0

    if not low < entry < high:
        raise argparse.ArgumentTypeError('Must be more than {} and less than {}'.format(low, high))

    return entry


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Script to prepare dataset for training and testing',
        epilog='You are welcome to contribute!',
    )

    parser.add_argument(
        '-p', '--path',
        type=check_is_dir,
        default=pathlib.Path.cwd(),
        required=False,
        help='Path to the directory with ASL dataset'
    )

    parser.add_argument(
        '-t', '--type',
        default='color',
        required=False,
        help='Type of images that would be used. Currently supported "color" and "depth"'
    )

    parser.add_argument(
        '-s', '--size',
        type=check_range,
        default=0.8,
        help='Size of train part'
    )

    return parser


def save(label_map: dict, data: dict, file_name: str):
    with pathlib.Path(file_name).open(mode='wt') as file:
        for class_id, images in data.items():
            if not label_map.get(class_id-1):
                raise ValueError('Label map does not contain letter {}'.format(class_id))

            for image in images:
                file.write('{} {}\n'.format(image, class_id-1))


def resize_images(train: dict, test: dict):
    mean, n = np.array([0, 0]), 0
    for _, images in train.items():
        for image in images:
            blob = cv2.imread(image)
            h, w, _ = blob.shape
            mean[0] += w
            mean[1] += h
            n += 1

    mean = np.round(mean / n).astype(np.int)

    def resize(data: dict):
        for _, paths in data.items():
            for path in paths:
                src = cv2.imread(path)
                dst = cv2.resize(src=src, dsize=(mean[0], mean[1]))
                cv2.imwrite(path, dst)

    print('Resizing to {}x{}'.format(mean[0], mean[1]))
    resize(train)
    resize(test)
    print('done')


def mean_image(train: dict):
    mean, n = None, 0
    for _, images in train.items():
        for image in images:
            n += 1
            blob = cv2.imread(image)
            if mean is not None:
                mean += blob
            else:
                mean = blob

    mean = mean / n
    blob = caffe.io.array_to_blobproto(mean)
    with open('mean_image.binaryproto', 'wb') as file:
        file.write(blob.SerializeToString())


def main():
    label_map = {
        0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
        9: 'a', 10: 'b', 11: 'c', 12: 'd', 13: 'e', 14: 'f', 15: 'g', 16: 'h',
        17: 'i', 18: 'k', 19: 'l', 20: 'm', 21: 'n', 22: 'o', 23: 'p', 24: 'q',
        25: 'r', 26: 's', 27: 't', 28: 'u', 29: 'x', 30: 'y'
    }

    with pathlib.Path('label_map.json').open(mode='wt') as file:
        json.dump(label_map, file)

    parser = build_argument_parser()
    arguments = parser.parse_args()

    data = {}
    for subject in arguments.path.iterdir():
        for file in subject.iterdir():
            if not file.name.endswith('.jpeg'):
                continue

            class_id = re.search('_C\\d\\d_', file.name)
            if not class_id:
                raise ValueError('can not find class id for image {}'.format(file))
            class_id = int(class_id.group()[2:-1])

            class_data = data.get(class_id, {})

            subject_data = class_data.get(subject.name, [])
            class_data.update({subject.name: subject_data + [file.as_posix()]})

            data.update({class_id: class_data})

    train_part = arguments.size

    train_images, test_images = {}, {}
    for class_id, subjects in data.items():
        train, test = [], []
        for subject, images in subjects.items():
            train_size = int(round(train_part * len(images)))
            assert 0 < train_size < len(images), 'Train and test images must be selected from each session of letters'

            shuffle(images)
            train.extend(images[:train_size])
            test.extend(images[train_size:])

        train_images.update({class_id: train})
        test_images.update({class_id: test})

    resize_images(train_images, test_images)
    #mean_image(train_images)

    save(label_map, train_images, 'train.lst')
    save(label_map, test_images, 'test.lst')

    return 0


if __name__ == '__main__':
    main()
