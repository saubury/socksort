#*****************************************************
# Sock Sort AWS DeepLens Inference Lambda Function
#
# Loads sock classification model, opens camera, and attempts to determine the most likely sock
#
# Refer to
# https://aws.amazon.com/blogs/machine-learning/build-your-own-object-classification-model-in-sagemaker-and-import-it-to-deeplens/
# https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-transfer-learning-highlevel.ipynb
# https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/object_detection_birds/object_detection_birds.ipynb

from threading import Thread, Event, Timer
import os
import json
import numpy as np
import greengrasssdk
import sys
import datetime
import time
import awscam
import cv2
import urllib
import zipfile
import mo
import mqttconfig
import paho.mqtt.client as mqtt


# Create a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and clould has a topic and a
# message body. This is the topic used to send messages to cloud.
iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

client.publish(topic=iot_topic, payload='At start of lambda function')

# external MQTT broker
client2 = mqtt.Client()
client2.username_pw_set(mqttconfig.mqtt_user, mqttconfig.mqtt_password)
client2.connect(mqttconfig.mqtt_broker_url, mqttconfig.mqtt_broker_port)

# Send to MQTT
client2.publish(topic=mqttconfig.mqtt_topic, payload='Hello from deeplens', qos=0, retain=False)

class LocalDisplay(Thread):
    # Class for facilitating the local display of inference results
    def __init__(self, resolution):
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()

def greengrass_infinite_infer_run():
    """ Entry point of the lambda function"""

    client.publish(topic=iot_topic, payload='Start of run loop...')

    try:
        # This object detection model 
        model_type = "classification"

        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()

        # The height and width of the training set images
        input_height = 512
        input_width  = 512

        # optimize the model
        client.publish(topic=iot_topic, payload='Optimizing model...')

        model_name = "image-classification"
        error, model_path = mo.optimize(model_name,input_width,input_height, aux_inputs={'--epoch': 15})
        # Load model to GPU (use {"GPU": 0} for CPU)
        mcfg = {"GPU": 1}
        model = awscam.Model(model_path, mcfg)

        client.publish(topic=iot_topic, payload='Custom object detection model loaded')

        # Load labels from text file
        with open('sock_labels.txt', 'r') as f:
	        labels = [l.rstrip() for l in f]
	   
        topk = 2

        # Send a starting message to IoT console
        client.publish(topic=iot_topic, payload="Inference is starting")

        doInfer = True
        while doInfer:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            # Raise an exception if failing to get a frame
            if ret == False:
                raise Exception("Failed to get frame from the stream")

            # Resize frame to fit model input requirement
            frameResize = cv2.resize(frame, (input_width, input_height))
        
            # Run model inference on the resized frame
            inferOutput = model.doInference(frameResize)

            # Output inference result to the fifo file so it can be viewed with mplayer
            parsed_results = model.parseResult(model_type, inferOutput)
            top_k = parsed_results[model_type][0:topk]
            
            sock_label = labels[top_k[0]["label"]]
            sock_prob = top_k[0]["prob"]*100
            
            # Write to MQTT
            json_payload = {"image" : sock_label, "probability" : sock_prob}
            client.publish(topic=iot_topic, payload=json.dumps(json_payload))
            client2.publish(topic=mqttconfig.mqtt_topic, payload=json.dumps(json_payload), qos=0, retain=False)

            # Write to image buffer;  screen display
            msg_screen = '{} {:.0f}%'.format(sock_label, sock_prob)
            cv2.putText(frame, msg_screen, (20,200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 12)
            local_display.set_frame_data(frame)            
            
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))

# Execute the function above
greengrass_infinite_infer_run()

# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return