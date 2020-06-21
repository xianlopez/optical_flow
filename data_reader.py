import os
import random
import cv2

class DataReader:
    # def __init__(self, kitti_path):
    #     self.index = 0
    #     self.path_pairs = []
    #     for scenario in os.listdir(kitti_path):
    #         print('Reading ' + scenario + '...')
    #         for day in os.listdir(os.path.join(kitti_path, scenario)):
    #             n_sequences = 0
    #             n_pairs = 0
    #             for drive in os.listdir(os.path.join(kitti_path, scenario, day)):
    #                 n_sequences += 1
    #                 images_dir = os.path.join(kitti_path, scenario, day, drive, 'image_02', 'data')
    #                 assert os.path.isdir(images_dir)
    #                 frames = os.listdir(images_dir)
    #                 n_pairs += len(frames) - 1
    #                 for i in range(len(frames) - 1):
    #                     self.path_pairs.append([os.path.join(images_dir, frames[i]),
    #                                             os.path.join(images_dir, frames[i + 1])])
    #             print('    Day ' + day + ': ' + str(n_pairs) + ' pairs in ' + str(n_sequences) + ' sequences.')
    #     print('Total number of pairs: ' + str(len(self.path_pairs)))
    #     # Randomize the image pairs:
    #     random.shuffle(self.path_pairs)

    def __init__(self, sintel_path):
        self.index = 0
        self.path_pairs = []
        sequences_folder = os.path.join(sintel_path, 'training', 'final')
        assert os.path.isdir(sequences_folder)
        for sequence in os.listdir(sequences_folder):
            images_dir = os.path.join(sequences_folder, sequence)
            frames = os.listdir(images_dir)
            for i in range(len(frames) - 1):
                self.path_pairs.append([os.path.join(images_dir, frames[i]),
                                        os.path.join(images_dir, frames[i + 1])])
        print('Total number of pairs: ' + str(len(self.path_pairs)))
        # Randomize the image pairs:
        random.shuffle(self.path_pairs)

    def GetNextBatch(self, batch_size):
        batch = []
        for i in range(batch_size):
            im1 = cv2.imread(self.path_pairs[self.index][0])
            im2 = cv2.imread(self.path_pairs[self.index][1])
            batch.append([im1, im2])
            self.index += 1
            if self.index >= len(self.path_pairs):
                self.index = 0
                random.shuffle(self.path_pairs)
                print('Rewinding data!')
        return batch


