#!/usr/bin/env python3
from __future__ import print_function




"""
 Copyright (c) Microsoft. All rights reserved.

 This code is licensed under the MIT License (MIT).
 THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
 ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
 IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
 PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
"""

""" Sample code to access HoloLens Research mode sensor stream """
# pylint: disable=C0103


import argparse
import socket
import sys
import binascii
import struct
from collections import namedtuple
import cv2
import numpy as np
from datetime import datetime

import numpy
import cv2
import cv2.aruco as aruco

PROCESS = True

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
# Definitions

# Protocol Header Format
# Cookie VersionMajor VersionMinor FrameType Timestamp ImageWidth
# ImageHeight PixelStride RowStride
SENSOR_STREAM_HEADER_FORMAT = "@IBBHqIIII"

SENSOR_FRAME_STREAM_HEADER = namedtuple(
    'SensorFrameStreamHeader',
    'Cookie VersionMajor VersionMinor FrameType Timestamp ImageWidth ImageHeight PixelStride RowStride'
)

# Each port corresponds to a single stream type
# Port for obtaining Photo Video Camera stream
PV_STREAM_PORT = 23940


def main(argv):
    """Receiver main"""
    parser = argparse.ArgumentParser()
    required_named_group = parser.add_argument_group('named arguments')

    required_named_group.add_argument("-a", "--host",
                                      help="Host address to connect", required=True)
    args = parser.parse_args(argv)
    message = 'hello there frend'
    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(tcp_echo_client(message, loop))
    ##################################
    # Create grid board object we're using in our stream
    board = aruco.GridBoard_create(
            markersX=2,
            markersY=2,
            markerLength=0.09,
            markerSeparation=0.01,
            dictionary=ARUCO_DICT)
    # Create vectors we'll be using for rotations and translations for postures
    rvecs, tvecs = None, None





    # Create a TCP Stream socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except (socket.error, msg):
        print("ERROR: Failed to create socket. Code: " + str(msg[0]) + ', Message: ' + msg[1])
        sys.exit()

    print('INFO: socket created')

    # Try connecting to the address
    s.connect((args.host, PV_STREAM_PORT))
    port=10000
    s2.connect((args.host,port))
    host2, port2='',10006
    s3.connect((host2,port2))
    
    print('INFO: Socket Connected to ' + args.host + ' on port ' + str(PV_STREAM_PORT))
    ################
    fps=1.2
    size=(1280,720)
    videoWriter = cv2.VideoWriter('MyOutput.avi', 
        cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)
    ######################
    file1 = open("myfile.txt","w") 
   
    # Load Yolo
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print(classes)
    layer_names = net.getLayerNames()
   
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print(output_layers)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Try receive data
    try:
        quit = False
        while not quit:
            tic= datetime.now()
            reply = s.recv(struct.calcsize(SENSOR_STREAM_HEADER_FORMAT))
            if not reply:
                print('ERROR: Failed to receive data')
                sys.exit()

            data = struct.unpack(SENSOR_STREAM_HEADER_FORMAT, reply)

            # Parse the header
            header = SENSOR_FRAME_STREAM_HEADER(*data)
            #print(header.ImageHeight,":",header.ImageWidth)
            # read the image in chunks
            image_size_bytes = header.ImageHeight * header.RowStride
            image_data = b''

            while len(image_data) < image_size_bytes:
                remaining_bytes = image_size_bytes - len(image_data)
                image_data_chunk = s.recv(remaining_bytes)
                if not image_data_chunk:
                    print('ERROR: Failed to receive image data')
                    sys.exit()
                image_data += image_data_chunk

            image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((header.ImageHeight,
                                        header.ImageWidth, header.PixelStride))
            if PROCESS:
                # process image
                image_array = cv2.cvtColor(image_array,cv2.cv2.COLOR_RGBA2RGB)
                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
                #image_array = cv2.Canny(gray,50,150,apertureSize = 3)

            #img = cv2.imread("room_ser.jpg")
            #width = 416
            #height = 416
            #dim = (width, height)
            #image_array = cv2.resize(image_array, dim, interpolation = cv2.INTER_AREA)
            height, width, channels = image_array.shape
            print("Width:"+str(width)+";Height:"+str(height))
            #print(channels)
            a=datetime.now() - tic
            #file1.writelines(a) 
            print("Reading images:"+str(a))
            #################################
            tica=datetime.now()
            blob = cv2.dnn.blobFromImage(image_array, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            ##########################
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            videoWriter.write(image_array)
            #asyncio.run(tcp_echo_client(str(len(boxes))))
            font = cv2.FONT_HERSHEY_PLAIN
            if len(boxes)==0:
                o="00:0"
                s2.sendall(o.encode("utf-8"))
            else:
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = colors[i]
                        cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(image_array, label, (x, y + 30), font, 3, color, 3)
                        
                        array="{}:{}:{}:{}:{}:{}:".format(class_ids[i],x*416/900,y*416/510,h*416/510,w*416/900,confidences[i])
                        #asyncio.run(tcp_echo_client(str(array)))
                        message=array
                        #asyncio.run(tcp_echo_client(message))
                        #data="1,2,3"
                 
                        #s.send("0 0 0 5 5 0.55")
                        s2.sendall(str(array).encode("utf-8"))
                        print(str(array).encode("utf-8"))
                        #data=s.recv(1024).decode("utf-8")
                        #print(data)
            # Detect Aruco markers
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        
            if ids is None:
                o="00:0"
                s3.sendall(o.encode("utf-8"))
            else:
               array2=""
               print(len(ids))
               for j in range(len(ids)):
                   id1=ids[j]
                   corner=corners[j]
                   # Print corners and ids to the console
                   print('ID: {}; Corners: {}'.format(id1, corner))
                   print('IDlength:{};Corners length:{}'.format(len(id1),len(corner)))
                   
                   if(j!=(len(ids)-1)):                      
                    array2="{}{}:{}:{}:{}:{}:{}:{}:{}:{}:".format(array2,id1[0],corner[0][0][0],corner[0][0][1],corner[0][1][0],corner[0][1][1],corner[0][2][0],corner[0][2][1],corner[0][3][0],corner[0][3][1])
                   else:
                    array2="{}{}:{}:{}:{}:{}:{}:{}:{}:{}".format(array2,id1[0],corner[0][0][0],corner[0][0][1],corner[0][1][0],corner[0][1][1],corner[0][2][0],corner[0][2][1],corner[0][3][0],corner[0][3][1])
                   #s3.sendall(str(array2).encode("utf-8"))  
                   print('Array2:{}'.format(array2))
                   # Outline all of the markers detected in our image
                   image_array = aruco.drawDetectedMarkers(image_array, corners, borderColor=(0, 0, 255))
               s3.sendall(str(array2).encode("utf-8")) 
            
            b=datetime.now() - tica
            #file1.writelines(b)
            print("For computing:"+str(b))
           
            cv2.imshow('Photo Video Camera Stream', image_array)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except KeyboardInterrupt:
        pass

    s.close()
    file1.close()
    cv2.destroyAllWindows()
    loop.close()



if __name__ == "__main__":
    main(sys.argv[1:])
