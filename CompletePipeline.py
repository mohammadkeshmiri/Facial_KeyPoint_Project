from data_load import Rescale, RandomCrop, Normalize, ToTensor
from data_load import FacialKeypointsDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Net


def Normalize(image):

    image_copy = np.copy(image)

    # convert image to grayscale
    image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # scale color range from [0, 255] to [0, 1]
    image_copy = image_copy/255.0

    return image_copy


def Rescale(image, output_size):

    h, w = image.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)

    img = cv2.resize(image, (new_w, new_h))

    return img


def RandomCrop(image, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    else:
        assert len(output_size) == 2
        output_size = output_size

    h, w = image.shape[:2]
    new_h, new_w = output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image = image[top: top + new_h,
                  left: left + new_w]

    return image


def ToTensor(image):
    """Convert ndarrays in sample to Tensors."""

       # if image has no grayscale color channel, add one
    if(len(image.shape) == 2):
        # add that third color dim
        image = image.reshape(1, image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((0, 3, 1, 2))

        return torch.from_numpy(image)


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0],
                predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')

# visualize the output
# by default this shows a batch of 10 images


def visualize_output(test_image, test_outputs, gt_pts=None):

    plt.figure(figsize=(20, 10))
    # un-transform the image data
    # image = test_image.numpy()   # convert to numpy array from a Tensor
    # transpose to go from torch to numpy image
    # image = np.transpose(image, (1, 2, 0))

    # un-transform the predicted key_pts data
    predicted_key_pts = test_outputs.detach().numpy()
     # undo normalization of keypoints
    predicted_key_pts = predicted_key_pts*50.0+100

      # plot ground truth points for comparison, if they exist
    ground_truth_pts = None
    if gt_pts is not None:
        ground_truth_pts = gt_pts[i]
        ground_truth_pts = ground_truth_pts*50.0+100

        # call show_all_keypoints
    show_all_keypoints(np.squeeze(test_image), np.squeeze(predicted_key_pts))
    plt.show()
    

# load in color image for face detection
image = cv2.imread('images/obamas.jpg')

# switch red and blue color channels
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
# # fig = plt.figure(figsize=(9,9))
# # plt.imshow(image)
# # plt.show()


# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier(
    'detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
images = []
for (x, y, w, h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    # cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)
    cropped_image = image_with_detections[y:y+h, x:x+w]
    images.append(cropped_image)

# fig = plt.figure(figsize=(9, 9))
# gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
# plt.imshow(gray, cmap='gray')
# plt.show()

# the transforms we defined in Notebook 1 are in the helper file `data_load.py`

model_dir = 'saved_models/'
model_name = 'keypoints_model.pt'
state_dict = torch.load(model_dir+model_name)

net = Net()
net.load_state_dict(state_dict)
net.eval()

transformed_Images = []
for img in images:
    rescaled = Rescale(img, 224)
    transformed_image = Normalize(rescaled)
    transformed_image = ToTensor(transformed_image)
    transformed_image = transformed_image.type(torch.FloatTensor)

    transformed_Images.append(transformed_image)

    predicted_key_pts = net(transformed_image)
    predicted_key_pts = predicted_key_pts.view(predicted_key_pts.size()[0], 68, -1)
    visualize_output(rescaled, predicted_key_pts)

