from google.cloud import vision
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import os
import io
import re

class LPDetector():

    def __init__(self, weights, config, names, output_dir, crop_dir):

        if not os.path.isdir(output_dir):
            os.mkdir('output')

        if not os.path.isdir(crop_dir):
            os.mkdir('cropped')

        self.output_dir = output_dir
        self.crop_dir = crop_dir

        self.frame = None

        # Initialize the parameters
        self.confThreshold = 0.5  #Confidence threshold
        self.nmsThreshold = 0.4  #Non-maximum suppression threshold

        self.inpWidth = 416  #608     #Width of network's input image
        self.inpHeight = 416 #608     #Height of network's input image

        # Load names of classes
        self.classesFile = names;

        self.classes = None
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        self.net = cv.dnn.readNetFromDarknet(config, weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Get the names of the output layers
    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self, file, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        cv.rectangle(self.frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cropImg = self.frame[top:bottom, left:right]
        cv.imwrite(os.path.join(self.crop_dir,'crop_'+file), cropImg)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(self.frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
        #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
        cv.putText(self.frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)


    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, file, outs):
        frameHeight = self.frame.shape[0]
        frameWidth = self.frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            print("out.shape : ", out.shape)
            for detection in out:
                #if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)
                #if scores[classId]>confThreshold:
                confidence = scores[classId]
                if detection[4]>self.confThreshold:
                    print(detection[4], " - ", scores[classId], " - th : ", self.confThreshold)
                    print(detection)
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(file, classIds[i], confidences[i], left, top, left + width, top + height)

    def detect_license_plate(self, dir):
        for file in os.listdir(dir):
            self.frame = cv.imread(os.path.join('test',file))

            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(self.frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)

            # Sets the input to the network
            self.net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = self.net.forward(self.getOutputsNames())

            # Remove the bounding boxes with low confidence
            self.postprocess(file, outs)

            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            t, _ = self.net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Write the frame with the detection boxes
            cv.imwrite(os.path.join(self.output_dir, file), self.frame.astype(np.uint8));
            print('Done')



class LPRecognizer():

    def __init__(self, credentials):
        # Set credentials as environment variable
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials

        self.client = vision.ImageAnnotatorClient()

        self.results = {}

    def read_license_plate(self, dir):

        for crop_file in os.listdir(dir):
            img_path = os.path.join(dir, crop_file)
            with io.open(img_path, 'rb') as image_file:
                    content = image_file.read()

            image = vision.types.Image(content=content)

            response = self.client.text_detection(image)

            try:
                num = re.sub('[^0-9a-zA-Z]','',response.text_annotations[0].description)
            except:
                pass

            self.results[img_path] = num

        return self.results
