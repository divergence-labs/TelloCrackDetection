import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
import tensorflow as tf
import pygame
import pygame.display
import pygame.key
import pygame.locals
import pygame.font
import os
import datetime
from subprocess import Popen, PIPE
from time import sleep

prev_flight_data = None
video_player = None
video_recorder = None
font = None
wid = None
date_fmt = '%Y-%m-%d_%H%M%S'

controls = {
    'w': 'forward',
    's': 'backward',
    'a': 'left',
    'd': 'right',
    'space': 'up',
    'left shift': 'down',
    'right shift': 'down',
    'q': 'counter_clockwise',
    'e': 'clockwise',
    # arrow keys for fast turns and altitude adjustments
    'left': lambda drone, speed: drone.counter_clockwise(speed*2),
    'right': lambda drone, speed: drone.clockwise(speed*2),
    'up': lambda drone, speed: drone.up(speed*2),
    'down': lambda drone, speed: drone.down(speed*2),
    'tab': lambda drone, speed: drone.takeoff(),
    'backspace': lambda drone, speed: drone.land(),
}


class FlightDataDisplay(object):
    # previous flight data value and surface to overlay
    _value = None
    _surface = None
    # function (drone, data) => new value
    # default is lambda drone,data: getattr(data, self._key)
    _update = None
    def __init__(self, key, format, colour=(255,255,255), update=None):
        self._key = key
        self._format = format
        self._colour = colour

        if update:
            self._update = update
        else:
            self._update = lambda drone,data: getattr(data, self._key)

    def update(self, drone, data):
        new_value = self._update(drone, data)
        if self._value != new_value:
            self._value = new_value
            self._surface = font.render(self._format % (new_value,), True, self._colour)
        return self._surface

def flight_data_mode(drone, *args):
    return (drone.zoom and "VID" or "PIC")

def flight_data_recording(*args):
    return (video_recorder and "REC 00:00" or "")  # TODO: duration of recording

def update_hud(hud, drone, flight_data):
    (w,h) = (158,0) # width available on side of screen in 4:3 mode
    blits = []
    for element in hud:
        surface = element.update(drone, flight_data)
        if surface is None:
            continue
        blits += [(surface, (0, h))]
        # w = max(w, surface.get_width())
        h += surface.get_height()
    h += 64  # add some padding
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0,0,0)) # remove for mplayer overlay mode
    for blit in blits:
        overlay.blit(*blit)
    pygame.display.get_surface().blit(overlay, (0,0))
    pygame.display.update(overlay.get_rect())

def status_print(text):
    pygame.display.set_caption(text)

hud = [
    FlightDataDisplay('height', 'ALT %3d'),
    FlightDataDisplay('ground_speed', 'SPD %3d'),
    FlightDataDisplay('battery_percentage', 'BAT %3d%%'),
    FlightDataDisplay('wifi_strength', 'NET %3d%%'),
    FlightDataDisplay(None, 'CAM %s', update=flight_data_mode),
    FlightDataDisplay(None, '%s', colour=(255, 0, 0), update=flight_data_recording),
]

def flightDataHandler(event, sender, data):
    global prev_flight_data
    text = str(data)
    if prev_flight_data != text:
        update_hud(hud, sender, data)
        prev_flight_data = text

def videoFrameHandler(event, sender, data):
    global video_player
    global video_recorder
    if video_player is None:
        cmd = [ 'mplayer', '-fps', '35', '-really-quiet' ]
        if wid is not None:
            cmd = cmd + [ '-wid', str(wid) ]
        video_player = Popen(cmd + ['-'], stdin=PIPE)

    try:
        video_player.stdin.write(data)
    except IOError as err:
        status_print(str(err))
        video_player = None

    try:
        if video_recorder:
            video_recorder.stdin.write(data)
    except IOError as err:
        status_print(str(err))
        video_recorder = None

def handleFileReceived(event, sender, data):
    global date_fmt
    # Create a file in ~/Pictures/ to receive image data from the drone.
    path = '%s/Pictures/tello-%s.jpeg' % (
        os.getenv('HOME'),
        datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    with open(path, 'wb') as fd:
        fd.write(data)
    status_print('Saved photo to %s' % path)


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = numpy.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

def main():
    pygame.init()
    pygame.display.init()
    pygame.display.set_mode((1280, 720))
    pygame.font.init()

    global font
    font = pygame.font.SysFont("dejavusansmono", 32)

    global wid
    if 'window' in pygame.display.get_wm_info():
        wid = pygame.display.get_wm_info()['window']
    print("Tello video WID:", wid)




    drone = tellopy.Tello()

    # drone.start_video()
    drone.subscribe(drone.EVENT_FLIGHT_DATA, flightDataHandler)
    # drone.subscribe(drone.EVENT_VIDEO_FRAME, videoFrameHandler)
    drone.subscribe(drone.EVENT_FILE_RECEIVED, handleFileReceived)
    speed = 30

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container = None
        model_path = '../../crak_graph/frozen_inference_graph.pb'
        odapi = DetectorAPI(path_to_ckpt=model_path)
        threshold = 0.6
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        # skip first 300 frames
        frame_skip = 0
        start = True

        droneTookOff = True
        while start:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                image = cv2.resize(image,(1920,1080))
                boxes, scores, classes, num = odapi.processFrame(image)
                for i in range(len(boxes)):
                    if classes[i] == 1 and scores[i] > threshold:
                        box = boxes[i]
                        cv2.rectangle(image, (box[1], box[0]), (box[3],box[2]), (255,0,0), 2)
                #cv2.putText(image, len(boxes) )
                cv2.imshow("Human", image)
                key = cv2.waitKey(1)
                print("val of key")
                print(key)
                time.sleep(0.01)  # loop with pygame.event.get() is too mush tight w/o some sleep
                for e in pygame.event.get():
                    # WASD for movement
                    if e.type == pygame.locals.KEYDOWN:
                        print('+' + pygame.key.name(e.key))
                        keyname = pygame.key.name(e.key)
                        if keyname == 'escape':
                            drone.quit()
                            exit(0)
                        if keyname in controls:
                            key_handler = controls[keyname]
                            if type(key_handler) == str:
                                getattr(drone, key_handler)(speed)
                            else:
                                key_handler(drone, speed)

                    elif e.type == pygame.locals.KEYUP:
                        print('-' + pygame.key.name(e.key))
                        keyname = pygame.key.name(e.key)
                        if keyname in controls:
                            key_handler = controls[keyname]
                            if type(key_handler) == str:
                                getattr(drone, key_handler)(0)
                            else:
                                key_handler(drone, 0)
                if key & 0xFF == ord('q'):
                    drone.land()
                    start = False
                    break
                # if key & 0xFF == ord('w'):
                #     print('w pressed')
                #     start = False
                #     drone.up(10)
                #     sleep(10)
                #
                #     start = True
                #     break
                # if key & 0xFF == ord('s'):
                #     print('s pressed')
                #     drone.down(10)
                #     key = -1
                #     break
                # if key & 0xFF == ord('a'):
                #     print('a pressed')
                #     drone.counter_clockwise(10)
                #
                #     break
                # if key & 0xFF== ord('d'):
                #     print('d pressed')
                #     drone.clockwise(10)
                #
                #     break
                # if key & 0xFF == ord('8'):
                #     print('8 pressed')
                #     drone.forward(10)
                #
                #     break
                # if key & 0xFF == ord('2'):
                #     print('2 pressed')
                #     drone.backward(10)
                #
                #     break
                # if key & 0xFF == ord('4'):
                #     print('4 pressed')
                #     drone.left(10)
                #
                #     break
                # if key & 0xFF == ord('6'):
                #     print('6 pressed')
                #     drone.right(10)
                #     sleep(5)
                #
                #     break
                # cv2.imshow('Original', image)
                # cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                # cv2.waitKey(1)
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                    print(time_base)
                else:
                    time_base = frame.time_base
                    print('else' + time_base)
                frame_skip = int((time.time() - start_time)/time_base) + 10
                key = -1
                print(frame_skip)
                drone.takeoff()

                # if(droneTookOff):
                #     drone.up(1)
                #     droneTookOff = False







    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        print('finally')
        drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
