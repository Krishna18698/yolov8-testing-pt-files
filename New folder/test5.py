sizeX=640
sizeY=640
import cv2
import numpy as np
import depthai
import blobconverter
import numpy as np
import json
import matplotlib.path as mplPath
from pathlib import Path
import sqlite3 as sl
import time
import argparse

def init_pipeline():
    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.createColorCamera()
    detection_nn = pipeline.createYoloDetectionNetwork()
    cam_rgb.setResolution(
        depthai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setInterleaved(True)

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    cam_rgb.setPreviewKeepAspectRatio(False)

    manip1 = pipeline.createImageManip()
    manip1.setMaxOutputFrameSize(1244160)
    manip1.initialConfig.setResize(sizeX, sizeY)
    cam_rgb.preview.link(manip1.inputImage)
    manip1.initialConfig.setFrameType(depthai.ImgFrame.Type.BGR888p)
    manip1.inputImage.setBlocking(True)

    if args.videoPath is not None:
        xinFrame = pipeline.create(depthai.node.XLinkIn)
        xinFrame.setStreamName("inFrame")
        xinFrame.out.link(manip1.inputImage)
        xinFrame.setMaxDataSize(1920*1080*3)
        nnPass = pipeline.create(depthai.node.XLinkOut)
        nnPass.setStreamName("pass")
        detection_nn.passthrough.link(xout_rgb.input)

    else:
        xinFrame = None

    # Extract the values from the JSON
    num_classes = config['nn_config']['NN_specific_metadata']['classes']
    coordinates = config['nn_config']['NN_specific_metadata']['coordinates']
    anchors = config['nn_config']['NN_specific_metadata']['anchors']
    anchor_masks = config['nn_config']['NN_specific_metadata']['anchor_masks']
    iou_threshold = config['nn_config']['NN_specific_metadata']['iou_threshold']

    # Set the values
    detection_nn.setNumClasses(num_classes)
    detection_nn.setCoordinateSize(coordinates)
    detection_nn.setAnchors(anchors)
    detection_nn.setAnchorMasks(anchor_masks)
    detection_nn.setIouThreshold(iou_threshold)
    detection_nn.setConfidenceThreshold(0.5)
    # detection_nn.setNumInferenceThreads(2)
    detection_nn.input.setBlocking(True)
    
    
    # Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
    # We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
    # detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
    # Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
    # Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
    manip1.out.link(detection_nn.input)

    if customModel is True:
        nnPath = str(
            (parentDir / Path('../../data/' + model)).resolve().absolute())
        # print(nnPath)
        detection_nn.setBlobPath(nnPath)
        print("Custom Model" + nnPath + "Size: " +
              str(sizeX) + "x" + str(sizeY))
    else:
        detection_nn.setBlobPath(blobconverter.from_zoo(
            name='person-detection-0106', shaves=6))
        print("Model from OpenVINO Zoo" + "Size: " +
              str(sizeX) + "x" + str(sizeY))

    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName("nn")
    detection_nn.out.link(xout_nn.input)
    return pipeline

def detect_and_count():
    global outputFrame, lock, zones_current_count, listeners, loop

    pipeline = init_pipeline()

    inputFrameShape = (sizeX, sizeY)

    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        q_nn = device.getOutputQueue("nn")
        

        # q_manip = device.getInputQueue("")

        baseTs = time.monotonic()
        simulatedFps = 30

        frame = None
        detections = []

        timestamp = datetime.utcnow()
        zone_data = []

        def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
            return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

        if args.videoPath is not None:
            videoPath = str(
                (parentDir / Path('../../data/' + video_source)).resolve().absolute())
            cap = cv2.VideoCapture(videoPath, cv2.CAP_FFMPEG)

        # loop over frames from the video stream
        while True:
            if args.videoPath is not None:
                read_correctly, frame = cap.read()
                qPass = device.getOutputQueue("pass")
                if not read_correctly:
                    break
                if args.videoPath is not None:
                    q_vid = device.getInputQueue(name="inFrame")
                    img = depthai.ImgFrame()
                    img.setType(depthai.ImgFrame.Type.BGR888p)
                    img.setData(to_planar(frame, inputFrameShape))
                    img.setTimestamp(baseTs)
                    baseTs += 1/simulatedFps

                    img.setWidth(inputFrameShape[0])
                    img.setHeight(inputFrameShape[1])
                    q_vid.send(img)
                    # in_vid = q_vid.tryGet()
                    print("hello", timestamp)
            if args.videoPath is not None:
                print("video")
                frame1 = qPass.get().getCvFrame()
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()
            
            if in_rgb is not None and args.videoPath is None:
                print("live")
                frame = in_rgb.getCvFrame()
            if in_nn is not None and args.videoPath is not None:
                print("vid")
                frame = frame1
                detections = in_nn.detections 
        

            if in_nn is not None and args.videoPath is None:
                print("detect")
                detections = in_nn.detections
            
            zone_data += check_overlap(frame, detections)
            print("done",timestamp)

            now = datetime.utcnow()
            if now.second != timestamp.second:
                t = threading.Thread(
                    target=insert_data, args=(zone_data, ))
                t.daemon = True
                t.start()
                zone_data = []
            timestamp = now

            with lock:
                outputFrame = frame.copy()
                print("finish")
            if args.videoPath is not None:
                ret, frame = cap.read()
                if not ret:
                    print("video over", timestamp)
                    cap.release()
                    break
                    # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
            if cv2.waitKey(1) == ord('q'):
                break

            
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--videoPath',
                        help="Path to video frame", default=None)

args = parser.parse_args()

video_source = args.videoPath
