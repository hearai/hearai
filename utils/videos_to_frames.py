import argparse
import cv2
import os
import threading
from queue import Queue

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Get frames as jpg files')
    parser.add_argument(
        '--path', metavar='DIR',
        help='path to base directory with video data',
        default='/dih4/dih4_2/hearai/data/korpus/labeled')
    parser.add_argument(
        '--dest', metavar='DES',
        help='path to destination directory to save frames',
        default='/dih4/dih4_2/hearai/data/frames')
    parser.add_argument(
        '-e', '--ext', 
        help='the file extension that the videos have',
        default='.mp4')
    parser.add_argument('-t', '--num-threads', type=int, default=30,
                        help='NUM_THREADS (default: 30)')
    return parser

def video_to_image(video_filename, out_dir):
    file_template = '{:s}_{:d}.jpg'
    reader = cv2.VideoCapture(video_filename)
    success, frame, = reader.read()  # read first frame
    folder_name = os.path.basename(out_dir)
    if success and not os.path.exists(out_dir):
        os.mkdir(out_dir)
    count = 0
    while success:
        out_filepath = os.path.join(out_dir,
                                    file_template.format(folder_name,
                                                         count))
        cv2.imwrite(out_filepath, frame)
        success, frame = reader.read()
        count += 1

def process_videofile(video_filename, video_path, image_out_path,
                      file_extension: str ='.mp4'):
    filepath = os.path.join(video_path, video_filename)
    video_filename = video_filename.replace(file_extension, '')

    out_dir = os.path.join(image_out_path, video_filename)

    video_to_image(filepath, out_dir)

def thread_job(queue, video_path, image_out_path, file_extension='.webm'):
    while not queue.empty():
        video_filename = queue.get()
        process_videofile(video_filename, video_path, image_out_path, file_extension=file_extension)
        queue.task_done()

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # the path to the folder which contains all video files
    filedir = args.path
    dataset_name = os.path.basename(filedir)

    # the root output path where RGB frame folders should be created
    dest_dir = args.dest
    image_out_path = os.path.join(dest_dir, dataset_name)
    if not os.path.exists(os.path.join(dest_dir)):
        os.mkdir(os.path.join(dest_dir))
    if not os.path.exists(image_out_path):
        os.mkdir(image_out_path)
    
    # the file extension that the videos have
    file_extension = args.ext

    video_filenames = os.listdir(filedir)
    queue = Queue()
    [queue.put(video_filename) for video_filename in video_filenames]

    NUM_THREADS = args.num_threads
    for i in range(NUM_THREADS):
        worker = threading.Thread(target=thread_job, args=(queue, filedir, image_out_path, file_extension))
        worker.start()

    print('Waiting for all videos to be completed.', queue.qsize(), 'videos')
    print('This can take a while depending on dataset size')
    queue.join()
    print('Done')
