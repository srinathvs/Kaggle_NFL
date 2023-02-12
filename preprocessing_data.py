# TODO : Add actual preprocessing steps for the dataset
import os

import numpy
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import pandas as pd


def create_frame_dataset_with_metadata(path_to_videos="D:\\Downloads\\Datasets\\nfl-player-contact-detection\\train\\",
                                       frame_directory="D:\\Downloads\Datasets\\nfl-player-contact-detection\\train_frames\\",
                                       video_metadata = "D:\\Downloads\\Datasets\\nfl-player-contact-detection\\train_video_metadata.csv",
                                       helmet_data = "D:\\Downloads\\Datasets\\nfl-player-contact-detection\\train_baseline_helmets.csv"):
    files = os.listdir(path_to_videos)

    if not os.path.exists(frame_directory):
        os.mkdir(frame_directory)

    video_metadata = pd.read_csv(video_metadata)
    helmet_metadata = pd.read_csv(helmet_data)


    for file in files:
        file_split = file.split('_')

        game_key = file_split[0]
        play_key = file_split[1]
        play_key = play_key.lstrip("0")
        type_key = file_split[2]
        type_key = type_key[:-4]

        helmet_slice = helmet_metadata.loc[(helmet_metadata['game_key'] == int(game_key)) & (helmet_metadata['play_id'] == int(play_key)) & (helmet_metadata['view'] == type_key)]

        if not helmet_slice.empty:
            base_file_name = os.path.basename(path_to_videos + file)
            if not os.path.exists(frame_directory + base_file_name + '/'):
                os.mkdir(frame_directory + base_file_name + '/')

            capture = cv2.VideoCapture(path_to_videos + file)
            relevant_frames = list(helmet_slice['frame'].unique())
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            out_path = frame_directory + base_file_name + '/'

            relevant_frame_pointer = 0
            frameNumber = 0

            while frameNumber < frame_count:
                frame_exists, frame = capture.read()
                if frame_exists and frameNumber == relevant_frames[relevant_frame_pointer]:
                    cv2.imshow('video', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                    cv2.imwrite(out_path + str(frameNumber) + '.png', frame)
                    relevant_frame_pointer += 1
                    if relevant_frame_pointer == len(relevant_frames):
                        break
                frameNumber += 1
            print(base_file_name + ' done\n')
            capture.release()



def test_helmet_bounding_boxes(frame_path, helmet_metadat_path):
    pass

if __name__ == '__main__':
    create_frame_dataset_with_metadata()
