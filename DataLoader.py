import numpy as np
import os
import struct
from six.moves import urllib
import gzip
import shutil

DATA_DIRECTORY = "data"
TRAIN_IMAGES_FILE = "train-images.idx3-ubyte"
TRAIN_LABELS_FILE = "train-labels.idx1-ubyte"
TEST_IMAGES_FILE = "t10k-images.idx3-ubyte"
TEST_LABELS_FILE = "t10k-labels.idx1-ubyte"

TRAIN_IMAGES_ZIP_FILE = "train-images-idx3-ubyte.gz"
TRAIN_LABELS_ZIP_FILE = "train-labels-idx1-ubyte.gz"
TEST_IMAGES_ZIP_FILE = "t10k-images-idx3-ubyte.gz"
TEST_LABELS_ZIP_FILE = "t10k-labels-idx1-ubyte.gz"


class MnistDataLoader:
    def __init__(self):
        self.prepare_data()

    def download_data(self, filename):
        """download mnist data from Yann Lecun's website"""
        DOWNLOAD_URL = "http://yann.lecun.com/exdb/mnist/"
        file_path = os.path.join(DATA_DIRECTORY, filename)
        if not os.path.exists(file_path):
            file_path, _ = urllib.request.urlretrieve(DOWNLOAD_URL + filename, file_path)
            stat = os.stat(file_path)
            print('Successfully downloaded....', filename, stat.st_size, 'bytes.')

    def extract_data(self, zip_file, file):
        zip_file_path = os.path.join(DATA_DIRECTORY, zip_file)
        file_path = os.path.join(DATA_DIRECTORY, file)
        with gzip.open(zip_file_path, 'rb') as input, open(file_path, 'wb') as output:
            shutil.copyfileobj(input, output)
        os.remove(zip_file_path)


    def load_image(self, file_name):
        file_path = os.path.join(DATA_DIRECTORY, file_name)
        with open(file_path, 'rb') as image_file:
            magic, num, rows, cols = struct.unpack(">IIII", image_file.read(16))
            images = np.fromfile(image_file, dtype=np.uint8).reshape(num, rows, cols)
            images = images.astype(np.float32)

        return images

    def load_labels(self, file_name):
        no_of_classes = 10
        file_path = os.path.join(DATA_DIRECTORY, file_name)
        with open(file_path, 'rb') as label_file:
            magic, num = struct.unpack(">II", label_file.read(8))
            labels = np.fromfile(label_file, dtype=np.int8)

        targets = labels.reshape(-1)
        labels = np.eye(no_of_classes)[targets]
        labels = labels.astype(np.float32)

        return labels

    def prepare_data(self):
        if not os.path.exists(DATA_DIRECTORY):
            os.mkdir(DATA_DIRECTORY)
        # If train images do not exist then download
        if not os.path.exists(os.path.join(DATA_DIRECTORY, TRAIN_IMAGES_FILE)):
            self.download_data(TRAIN_IMAGES_ZIP_FILE)
            self.extract_data(TRAIN_IMAGES_ZIP_FILE, TRAIN_IMAGES_FILE)
        # If train labels do not exist then download
        if not os.path.exists(os.path.join(DATA_DIRECTORY, TRAIN_LABELS_FILE)):
            self.download_data(TRAIN_LABELS_ZIP_FILE)
            self.extract_data(TRAIN_LABELS_ZIP_FILE, TRAIN_LABELS_FILE)
        # If test images do not exist then download
        if not os.path.exists(os.path.join(DATA_DIRECTORY, TEST_IMAGES_FILE)):
            self.download_data(TEST_IMAGES_ZIP_FILE)
            self.extract_data(TEST_IMAGES_ZIP_FILE, TEST_IMAGES_FILE)
        # If test labels do not exist then download
        if not os.path.exists(os.path.join(DATA_DIRECTORY, TEST_LABELS_FILE)):
            self.download_data(TEST_LABELS_ZIP_FILE)
            self.extract_data(TEST_LABELS_ZIP_FILE, TEST_LABELS_FILE)

    def load_all_train_data(self):
        train_images = self.load_image(TRAIN_IMAGES_FILE)
        train_labels = self.load_labels(TRAIN_LABELS_FILE)

        return (train_images, train_labels)

    def load_all_test_data(self):
        test_images = self.load_image(TEST_IMAGES_FILE)
        test_labels = self.load_labels(TEST_LABELS_FILE)

        return (test_images, test_labels)

    def load_train_batch(self, start_index, batch_size):
        train_images, train_labels = self.load_all_train_data()

        end_index = start_index + batch_size
        if end_index > len(train_labels):
            end_index = len(train_labels)

        return (train_images[start_index:end_index], train_labels[start_index:end_index])

    def load_test_batch(self, start_index, batch_size):
        test_images, test_labels = self.load_all_test_data()

        end_index = start_index + batch_size
        if end_index > len(test_labels):
            end_index = len(test_labels)

        return (test_images[start_index:end_index], test_labels[start_index:end_index])