# Catosight
Catosight is a Python application that uses a Kinect sensor connected to a PC to identify and catalog items in a garage. This application captures images of items using the Kinect sensor, identifies the items using a pre-trained machine learning model, and prints the predicted class of each item.

Installation
Before running Catosight, you need to install several dependencies. These include PyTorch, torchvision, OpenCV, and PyKinect2. You can install these dependencies using pip:

bash
Copy code
pip install torch torchvision torchaudio cudatoolkit=11.1 opencv-python pykinect2
This command installs PyTorch with support for CUDA 11.1. Replace 11.1 with the version of CUDA installed on your machine.

You'll also need a file called imagenet-simple-labels.json, which contains the names of the classes that the machine learning model can recognize. You can download this file from [here.](https://github.com/anishathalye/imagenet-simple-labels/blob/master/imagenet-simple-labels.json)

Usage
To run Catosight, execute the main Python script:

bash
Copy code
python catosight.py
The script will capture a specified number of frames from the Kinect sensor, save them as PNG files, and then pass each image through a pre-trained MobileNetV2 model to identify the most likely object in the image. The name of the predicted class for each image will be printed to the console.
