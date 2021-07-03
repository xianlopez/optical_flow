import os
import subprocess

drive_path = r'C:\datasets\DrivingInBristol\drive_06_27'
assert os.path.isdir(drive_path)

for name in os.listdir(drive_path):
    if name[-4:] == '.MTS':
        print('Processing ' + name)
        rawname = name[:-4]
        video_path = os.path.join(drive_path, name)
        shot_dir = os.path.join(drive_path, rawname)
        os.makedirs(shot_dir)
        out_path = os.path.join(shot_dir, '%04d.png')
        subprocess.call(
            ['ffmpeg', '-i', video_path, '-vf', 'yadif', '-r', '10', '-s', '960:540', out_path])




