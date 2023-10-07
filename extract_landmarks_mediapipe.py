"""
export LD_LIBRARY_PATH=/home/lasii/anaconda3/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_DIR=/usr/lib/cuda
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_DIR}
"""
import cv2
import mediapipe as mp
import pandas as pd
import os
import glob
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

ROOT = 'landmarks_mediapipe'
def get_frames(filename):
    video=cv2.VideoCapture(filename)
    while video.isOpened():
        rete,frame = video.read()
        if rete:
            yield frame
        else:
            break
    video.release()
    
def sample_frames(frames, max_frames=300):
    if len(frames) <= max_frames:
        return frames
    hops = len(frames)//max_frames
    sampled = []
    
    for i in range(0, len(frames), hops):
        sampled.append(frames[i])
        if len(sampled) == max_frames:
            return sampled
        
def parse_landmarks(results, fno):
    landmarks = {}
    landmarks['frame'] = fno
    for idx, l in enumerate(results.pose_landmarks.ListFields()[0][1]):
        landmarks[f'x{idx}']= l.x 
        landmarks[f'y{idx}']= l.y
        landmarks[f'z{idx}']= l.z
        landmarks[f'visibility{idx}']= l.visibility
    
    return landmarks


def extract_landmarks(vid_path):
    """
    Extracts raw landmarks from video, using mediapipe
    """
    frames = list(get_frames(vid_path))
    frames = sample_frames(frames)
    print (f"# Frames: {len(frames)}")
    vid_landmarks = []
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        for idx, image in enumerate(frames):
#             print (f"Frame: {idx}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            vid_landmarks.append(parse_landmarks(results, idx))
    return pd.DataFrame().from_records(vid_landmarks)

def get_files_list():
    files = glob.glob('/home/lasii/ximi/deepu1/chunks/*.mp4')
    with open('processed_landmarks.txt', 'r') as f:
        processed = [i.strip('\n') for i in f.readlines()]
    return list(set(files) - set(processed))

def extract_and_save(root, vid_path):
    df = extract_landmarks(vid_path)
    fname = vid_path.strip('.mp4').split('/')[-1] + '.csv'
    path = os.path.join(root, fname)
    print (f"Saving to {path} ...")
    df.to_csv(path, index=False)
    
def main():
    files = get_files_list()
    for vid_path in tqdm(files):
        print (f"File name: {vid_path}")
        try:
            extract_and_save(ROOT, vid_path)
            with open('processed_landmarks.txt', 'a') as pr:
                pr.write(vid_path)
                pr.write('\n')
                
        except Exception as e:
            print (e)
            with open('errors_landmarks.txt', 'a') as err:
                err.write(vid_path)
                err.write('\n')
if __name__ == '__main__':
    main()