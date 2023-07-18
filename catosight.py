import cv2
import json
import torch
from PIL import Image
from torchvision import models, transforms
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime

# Load the pre-trained model
model = models.mobilenet_v2(pretrained=True)

# Move the model to GPU
model = model.to('cuda')

# Set the model to evaluation mode
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the class names
with open('imagenet-simple-labels.json') as f:
    labels = json.load(f)

def class_id_to_label(i):
    return labels[i]

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to('cuda')
    with torch.no_grad():
        output = model(image)
    _, predicted_idx = torch.max(output, 1)
    predicted_class = class_id_to_label(predicted_idx.item())
    return predicted_class

# Initialize Kinect
kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
color_width, color_height = kinect.color_frame_desc.Width, kinect.color_frame_desc.Height
depth_width, depth_height = kinect.depth_frame_desc.Width, kinect.depth_frame_desc.Height

# Capture and process images
frames_to_capture = 10
frames_captured = 0

while frames_captured < frames_to_capture:
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape((color_height, color_width, 4))
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite('frame_{}.png'.format(frames_captured), color_frame)
        frames_captured += 1

kinect.close()

# Predict object in images
for i in range(frames_to_capture):
    image_path = 'frame_{}.png'.format(i)
    predicted_class = predict_image(image_path)
    print('Image {}: {}'.format(i, predicted_class))
