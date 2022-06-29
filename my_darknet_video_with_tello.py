from ctypes import *
import random
import os
from typing import Container
from typing_extensions import final
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue

#add import for tello
import tellopy
import math
import sys
import traceback
import av
import numpy


def parser():
    parser = argparse.ArgumentParser(description="YOLOの物体検出")
    parser.add_argument("--input", type=str, default=0,
                        help="ビデオソース。空の場合、Webカメラ(0)のストリームを使用")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inferenceのファイル名。空の場合は保存されない")
    parser.add_argument("--weights", default="C:/Hidaka/darknet/build/darknet/x64/myfile/learning_final_raven_human.weights",
                        help="weightsファイルのパス")
    parser.add_argument("--dont_show", action='store_true',
                        help="inferenceウィンドウの表示取り消し。組み込みシステム用")
    parser.add_argument("--ext_output", action='store_true',
                        help="検出されたオブジェクトのbbox座標を表示するかどうか")
    parser.add_argument("--config_file", default="C:/Hidaka/darknet/build/darknet/x64/myfile/learning.cfg",
                        help="configファイルのパス")
    parser.add_argument("--data_file", default="C:/Hidaka/darknet/build/darknet/x64/myfile/learning.data",
                        help="dataファイルのパス")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="確からしさの閾値。この値より低い検出物を取り除く")
    return parser.parse_args()


def drone_halt():
    drone.forward(0)
    drone.backward(0)
    drone.right(0)
    drone.left(0)
    drone.up(0)
    drone.down(0)


def str2int(video_path):
    """
    argparseは文字列を返しますが、ウェブカメラはint(0, 1 ...)を使用。
    必要に応じてint型にキャスト。
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
                                #しきい値は、0から1の間の浮動小数点でなければならない（非包含）。信頼度がこの値よりも低い検出物を取り除く
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLOフォーマットでは、アノテーションに相対座標を使用。
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue, container):
    """
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),interpolation=cv2.INTER_LINEAR)
            frame_queue.put(frame)
            img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
            darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
            darknet_image_queue.put(img_for_detect)
        cap.release()
    """
    global flag
    frame_skip = 300
    while flag == True:
        for cont in container.decode(video = 0):
            #print("videocaputure complete")
            if 0 < frame_skip:
                frame_skip -= 1
                continue
            start_time = time.time()
            frame = cv2.cvtColor(numpy.array(cont.to_image()), cv2.COLOR_RGB2BGR)

            if cont.time_base < 1.0 / 6.0:
                time_base = 1.0 / 6.0
            else:
                time_base = cont.time_base
            frame_skip = int((time.time() - start_time) / time_base)

            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),interpolation = cv2.INTER_LINEAR)
            frame_resized = cv2.resize(frame, (darknet_width, darknet_height),interpolation = cv2.INTER_LINEAR)
            frame_queue.put(frame_resized)
            img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
            darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
            darknet_image_queue.put(img_for_detect)


def inference(darknet_image_queue, detections_queue, fps_queue):
    global direction
    global flag
    global i
    global fps_ave
    global fps_max
    global detection_raven
    global detection_not

    while flag == True:
        print("-----------------------------------")
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        if (time.time() - start_time) >= 60:
            fps_ave *= i
            i += 1
            fps_ave = (fps_ave + fps) / i
            if fps > fps_max:
                fps_max = fps
            print("AVERAGE {}".format(fps_ave))
            print("MAX {}".format(fps_max))
        else:
            fps_ave = fps
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)

#認識維持率実験用
        if (time.time() - start_time) >= 60:
            if not detections:
                detection_not += 1
            else:
                for obj in detections:
                    if obj[0] == "raven":
                        detection_raven += 1
                    else:
                        detection_not += 1
            print("detection raven percentage:{}".format((detection_raven / (detection_raven + detection_not)) * 100))
            print("detection_raven percentage:{}".format(detection_raven))
            print("time:{}".format(time.time() - (start_time - 60)))
            drone_halt()


        #以下追尾用制御
        if not detections:
            print("not detection!")
            drone_halt()
        else:
            for obj in detections:
                if obj[0] == "raven":
                    bbox = [0, 0, 0, 0]
                    bbox[0] = obj[2][0] #x座標
                    bbox[1] = obj[2][1] #y座標
                    bbox[2] = obj[2][2] #横 w
                    bbox[3] = obj[2][3] #縦 h

                    if bbox[0] < 240 and bbox[3] < 240:
                        drone.forward(80)
                        print("forward!")
                    elif bbox[0] > 310  and bbox[3] > 310:
                        drone.backward(80)
                        print("backward!")
                    else:
                        drone.forward(0)
                        print("halted! (front and back)")

                    #画像サイズは416×416
                    if bbox[0] < 158:
                        direction = "left"
                        drone.left(80)
                        print("left!")
                    elif bbox[0] > 258:
                        direction = "right"
                        drone.right(80)
                        print("right!")
                    else:
                        drone.left(0)
                        print("halted! (rabbit horns)")

                    if bbox[1] < 158:
                        drone.up(80)
                        print("upped!")
                    elif bbox[1] > 258:
                        drone.down(80)
                        print("downed!")
                    else:
                        drone.up(0)
                        print("halted (up and down)")

                else:
                    print("can't find the bird!")
                    print(direction)
                    drone_halt()

                    if direction == "right":
                        drone.clockwise(20)
                        print("searching! (clockwise)")
                        #print()
                    elif direction == "left":
                        drone.counter_clockwise(20)
                        print("searching! (counter_clockwise)")
                        #print()


def drawing(frame_queue, detections_queue, fps_queue):
    global flag
    random.seed(3)  # deterministic bbox colors
    #video = set_saved_video(cap, args.out_filename, (video_width, video_height))
    while flag == True:
        frame = frame_queue.get()

        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))

            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            if not args.dont_show:
                cv2.imshow('Inference', image)
                cv2.waitKey(1)

            #if args.out_filename is not None:
            #    video.write(image)
            if cv2.waitKey(10) == 27:
                flag = False
                drone.land()
                drone.quit()
                break


if __name__ == '__main__':
    #darknetの準備
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    direction = "none"
    flag = True
    i = 1
    fps_ave = 0
    fps_max = 0
    start_time = time.time()
    detection_raven = 0
    detection_not = 0


    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    #darknetはnumpyのフォーマット画像を受け付けない。
    #各検出器に再利用するイメージで作成する
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)

    #以下Teloの準備
    drone = tellopy.Tello()

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
                print("container is opened")
            except av.AVError as ave:
                print(ave)
                print("retrying...")

        #接続完了
        #最初の300フレームは飛ばす
        #frame_skip = 300
        #離陸
        #drone.takeoff()

        video_width = 1280
        video_height = 720
        a = Thread(target=video_capture, args=(frame_queue, darknet_image_queue, container))
        b = Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue))
        c = Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue))

        a.setDaemon(True)
        b.setDaemon(True)
        c.setDaemon(True)

        a.start()
        b.start()
        c.start()

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)