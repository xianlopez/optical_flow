import os
import random
import cv2

def read_kitti(kitti_path):
    path_pairs = []
    for scenario in os.listdir(kitti_path):
        print('Reading ' + scenario + '...')
        for day in os.listdir(os.path.join(kitti_path, scenario)):
            n_sequences = 0
            n_pairs = 0
            for drive in os.listdir(os.path.join(kitti_path, scenario, day)):
                n_sequences += 1
                images_dir = os.path.join(kitti_path, scenario, day, drive, 'image_02', 'data')
                assert os.path.isdir(images_dir)
                frames = os.listdir(images_dir)
                frames.sort()
                n_pairs += len(frames) - 1
                for i in range(len(frames) - 1):
                    path_pairs.append([os.path.join(images_dir, frames[i]),
                                            os.path.join(images_dir, frames[i + 1])])
            print('    Day ' + day + ': ' + str(n_pairs) + ' pairs in ' + str(n_sequences) + ' sequences.')
    print('Total number of KITTI pairs: ' + str(len(path_pairs)))
    return path_pairs

def read_sintel(sintel_path):
    path_pairs = []
    sequences_folder = os.path.join(sintel_path, 'training', 'final')
    assert os.path.isdir(sequences_folder)
    for sequence in os.listdir(sequences_folder):
        images_dir = os.path.join(sequences_folder, sequence)
        frames = os.listdir(images_dir)
        frames.sort()
        for i in range(len(frames) - 1):
            path_pairs.append([os.path.join(images_dir, frames[i]),
                                    os.path.join(images_dir, frames[i + 1])])
    print('Total number of Sintel pairs: ' + str(len(path_pairs)))
    return path_pairs

def read_youtube(youtube_path):
    path_pairs = []
    for video in os.listdir(youtube_path):
        print('Reading video ' + video)
        n_shots = 0
        n_pairs = 0
        for shot in os.listdir(os.path.join(youtube_path, video)):
            shot_dir = os.path.join(youtube_path, video, shot)
            if os.path.isdir(shot_dir):
                n_shots += 1
                frames = os.listdir(shot_dir)
                frames.sort()
                for i in range(len(frames) - 1):
                    path_pairs.append([os.path.join(shot_dir, frames[i]),
                                            os.path.join(shot_dir, frames[i + 1])])
                    n_pairs += 1
        print(str(n_pairs) + ' pairs in ' + str(n_shots) + ' shots')
    print('Total number of Youtube pairs: ' + str(len(path_pairs)))
    return path_pairs

class DataReader:
    def __init__(self, kitti_path, sintel_path, youtube_path):
        self.index = 0
        self.path_pairs = []
        if kitti_path:
            self.path_pairs.extend(read_kitti(kitti_path))
        if sintel_path:
            self.path_pairs.extend(read_sintel(sintel_path))
        if youtube_path:
            self.path_pairs.extend(read_youtube(youtube_path))
        # Randomize the image pairs:
        random.shuffle(self.path_pairs)
        print("Total number of pairs: " + str(len(self.path_pairs)))

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


