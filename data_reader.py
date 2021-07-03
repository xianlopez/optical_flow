from multiprocessing import Pool, Queue
import numpy as np
import os
import cv2
import random

image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])


def read_kitti(kitti_path):
    path_pairs = []
    for day in os.listdir(os.path.join(kitti_path)):
        for drive in os.listdir(os.path.join(kitti_path, day)):
            if drive[-4:] == '.txt':
                continue
            images_dir_l = os.path.join(kitti_path, day, drive, 'image_02', 'data')
            images_dir_r = os.path.join(kitti_path, day, drive, 'image_03', 'data')
            assert os.path.isdir(images_dir_l)
            assert os.path.isdir(images_dir_r)
            frames = os.listdir(images_dir_l)
            frames.sort()
            for i in range(len(frames) - 1):
                path_pairs.append([os.path.join(images_dir_l, frames[i]),
                                   os.path.join(images_dir_l, frames[i + 1])])
            for i in range(len(frames)):
                assert os.path.isfile(os.path.join(images_dir_r, frames[i]))
                path_pairs.append([os.path.join(images_dir_l, frames[i]),
                                   os.path.join(images_dir_r, frames[i])])
    print('Total number of KITTI pairs: ' + str(len(path_pairs)))
    return path_pairs


def read_batch(batch_info, opts):
    batch_imgs_np = np.zeros((opts.batch_size, opts.img_height, opts.img_width, 6), np.float32)
    for i in range(len(batch_info)):
        item_info = batch_info[i]
        image1, image2 = read_item(item_info, opts)
        batch_imgs_np[i, :, :, :3] = image1
        batch_imgs_np[i, :, :, 3:] = image2
    output_queue.put(batch_imgs_np)


def read_item(item_info, opts):
    if np.random.rand() < 0.5:
        path1 = item_info[0]
        path2 = item_info[1]
    else:
        path1 = item_info[1]
        path2 = item_info[0]
    # Read images:
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # Resize:
    img1 = cv2.resize(img1, (opts.img_width, opts.img_height))
    img2 = cv2.resize(img2, (opts.img_width, opts.img_height))
    # Make pixel values between 0 and 1:
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    # Subtract mean:
    img1 = img1 - image_means
    img2 = img2 - image_means
    return img1, img2


def init_worker(queue):
    global output_queue
    output_queue = queue


class ReaderOpts:
    def __init__(self, kitti_path, batch_size, img_height, img_width, nworkers):
        self.kitti_path = kitti_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.nworkers = nworkers


class AsyncReader:
    def __init__(self, opts):
        self.opts = opts
        self.data_info = read_kitti(opts.kitti_path)
        # self.data_info = self.data_info[:200]
        self.nbatches = len(self.data_info) // opts.batch_size

        self.output_queue = Queue()
        self.pool = Pool(processes=self.opts.nworkers, initializer=init_worker, initargs=(self.output_queue,))
        self.next_batch_idx = 0
        random.shuffle(self.data_info)
        for i in range(min(self.opts.nworkers, self.nbatches)):
            self.add_fetch_task()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        print('Closing AsyncReader')
        self.pool.close()
        # OpenCV seems to play bad with multiprocessing, so I need to add this here. Maybe I could change
        # the reading of the images to use skimage instead of cv2.
        self.pool.terminate()
        self.pool.join()
        print('Closed')

    def add_fetch_task(self):
        batch_info = []
        for i in range(self.opts.batch_size):
            batch_info.append(self.data_info[self.next_batch_idx * self.opts.batch_size + i])
        self.pool.apply_async(read_batch, args=(batch_info, self.opts))
        if self.next_batch_idx == self.nbatches - 1:
            self.next_batch_idx = 0
            random.shuffle(self.data_info)
        else:
            self.next_batch_idx += 1

    def get_batch(self):
        imgs = self.output_queue.get()
        self.add_fetch_task()
        return imgs

