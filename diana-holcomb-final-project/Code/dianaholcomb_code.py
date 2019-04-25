import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
import geojson
import random
import os
import json
import time

# -----------------------------------------------------------------------------------
## Get the list of labels
#
# Load the class number -> class string label map
labels_filepath = './xView_baseline/xview_class_labels.txt'


def get_labels(labels_filepath):
    labels = {}
    with open(labels_filepath) as classfile:
        data = classfile.readlines()
        for line in data:
            if len(line) > 0:
                class_num, class_name = line.split(':')
                labels[class_num] = class_name.strip()
    return labels

labels = get_labels(labels_filepath)
print(labels)
print("Num labels: " + str(len(labels)))

# -----------------------------------------------------------------------------------
### The code in the cell below is the official xView preprocessing code.
#
# Found here: https://github.com/DIUx-xView/data_utilities/blob/master/wv_util.py
# Some mods were made to get_labels to allow a filter to break up test and train.

def scale(x, range1=(0, 0), range2=(0, 0)):
    """
    Linear scaling for a value x
    """
    return range2[0] * (1 - (x - range1[0]) / (range1[1] - range1[0])) + range2[1] * (
                (x - range1[0]) / (range1[1] - range1[0]))


def get_image(fname):
    """
    Get an image from a filepath in ndarray format
    """
    img = torch.from_numpy(np.asarray(Image.open(fname)))
    return img


def get_labels(fname, filtered_list=None):
    """
    Gets label data from a geojson label file

    Args:
        fname: file path to an xView geojson label file

    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    """
    with open(fname) as f:
        data = json.load(f)

    if filtered_list != None:
        print("Pre-filtered geojson feature length: " + str(len(data['features'])))
        # Filter python objects with list comprehensions
        data['features'] = [feature for feature in data['features'] if
                            not feature['properties']['image_id'] in filtered_list]
        print("Post-filtered geojson feature length: " + str(len(data['features'])))

    coords = np.zeros((len(data['features']), 4))
    chips = np.zeros((len(data['features'])), dtype="object")
    classes = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            # if (filtered_list == None) or (filtered_list != None and b_id in filtered_list):
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val

    return coords, chips, classes


def boxes_from_coords(coords):
    """
    Processes a coordinate array from a geojson into (xmin,ymin,xmax,ymax) format

    Args:
        coords: an array of bounding box coordinates

    Output:
        Returns an array of shape (N,4) with coordinates in proper format
    """
    nc = np.zeros((coords.shape[0], 4))
    for ind in range(coords.shape[0]):
        x1, x2 = coords[ind, :, 0].min(), coords[ind, :, 0].max()
        y1, y2 = coords[ind, :, 1].min(), coords[ind, :, 1].max()
        nc[ind] = [x1, y1, x2, y2]
    return nc


def chip_image(img, coords, classes, shape=(300, 300)):
    """
    Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
        multiple chips are clipped: each portion that is in a chip is labeled. For example,
        half a building will be labeled if it is cut off in a chip. If there are no boxes,
        the boxes array will be [[0,0,0,0]] and classes [0].
        Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.

    Args:
        img: the image to be chipped in array format
        coords: an (N,4) array of bounding box coordinates for that image
        classes: an (N,1) array of classes for each bounding box
        shape: an (W,H) tuple indicating width and height of chips

    Output:
        An image array of shape (M,W,H,C), where M is the number of chips,
        W and H are the dimensions of the image, and C is the number of color
        channels.  Also returns boxes and classes dictionaries for each corresponding chip.
    """

    height, width, channels = img.shape
    wn, hn = shape

    w_num, h_num = (int(width / wn), int(height / hn))
    images = np.zeros((w_num * h_num, hn, wn, channels))
    total_boxes = {}
    total_classes = {}

    k = 0
    for i in range(w_num):
        for j in range(h_num):
            x = np.logical_or(np.logical_and((coords[:, 0] < ((i + 1) * wn)), (coords[:, 0] > (i * wn))),
                              np.logical_and((coords[:, 2] < ((i + 1) * wn)), (coords[:, 2] > (i * wn))))
            out = coords[x]
            y = np.logical_or(np.logical_and((out[:, 1] < ((j + 1) * hn)), (out[:, 1] > (j * hn))),
                              np.logical_and((out[:, 3] < ((j + 1) * hn)), (out[:, 3] > (j * hn))))
            outn = out[y]
            out = np.transpose(np.vstack((np.clip(outn[:, 0] - (wn * i), 0, wn),
                                          np.clip(outn[:, 1] - (hn * j), 0, hn),
                                          np.clip(outn[:, 2] - (wn * i), 0, wn),
                                          np.clip(outn[:, 3] - (hn * j), 0, hn))))
            box_classes = classes[x][y]

            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0, 0, 0, 0]])
                total_classes[k] = np.array([0])

            chip = img[hn * j:hn * (j + 1), wn * i:wn * (i + 1), :channels]
            images[k] = chip

            k = k + 1

    return images.astype(np.uint8), total_boxes, total_classes

# -----------------------------------------------------------------------------------
## Image Preprocessing
#
# First we must be able to break the image up into chips.

def preprocess_dataset(original_image_loc, groundtruth_filepath, chip_size, limit=None):
    # first process the geojson
    coords, chips, classes = get_labels(groundtruth_filepath)
    chip_bytes = []
    chip_coords = []
    chip_labels = []
    chip_orig_image_names = []
    image_names = [f for f in os.listdir(original_image_loc) if os.path.isfile(os.path.join(original_image_loc, f))]
    print("Number of images to preprocess: " + str(len(image_names)))
    count = 0;
    for _file in image_names:
        if limit != None and count < limit:
            c_img, c_box, c_cls = chip_image(get_image(original_image_loc + _file), coords, classes,
                                             shape=(chip_size, chip_size))
            for i in range(len(c_img)):
                chip_bytes.append(c_img[i])
                chip_coords.append(c_box[i])
                chip_labels.append(c_cls[i])
                chip_orig_image_names.append(_file)
            print(".", end="")
        count = count + 1

    print("Done with preprocessing.")
    return chip_bytes, chip_coords, chip_labels, chip_orig_image_names

# -----------------------------------------------------------------------------------
## Create a custom PyTorch Dataset to load the xview files

class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


class xViewDataset(Dataset):
    # input is image, target is annotation
    def __init__(self, image_arr, coord_arr, label_arr,
                 transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.images = image_arr
        print("Len of img array: " + str(len(self.images)))
        # filter the giant geojso data into just the images included in the root folder
        self.coords = coord_arr
        self.labels = label_arr

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.images)  # self.images.shape[0]

    def pull_item(self, index):
        img = self.images[index]
        height, width, channels = img.shape

        target = self.labels[index]
        if self.target_transform is not None:
            # convert the bbox and label into one
            target = self.target_transform(self.coords[index], np.array(self.labels[index], ndmin=2).T, height, width)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        image = self.images[index]
        return PIL.Image.fromarray(image, mode="RGB")

    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = str(index)
        gt = self.target_transform(self.boxes[index], self.labels[index], 1, 1)

        return img_id, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

# -----------------------------------------------------------------------------------
### IMPORTANT INITIAL VALUES USED IN ALL CODE BELOW

chip_size = 300
in_channels = 3
input_size = chip_size**2
num_classes = len(labels)
num_epochs = 100
batch_size = 10
learning_rate = 0.001
train_path = './xView/train_images/'
test_path = './xView/val_images/'
geojson_path = './xView/xView_train.geojson'
kernel_size = 5
conv_size = chip_size

# -----------------------------------------------------------------------------------
## Choose whether to preprocess the raw data, or load the pre-saved pickle files:
#
# * train_chip_bytes = 'train_chip_bytes.pkl'
#   * an array of the bytes of all the chipped images.
#   * Size: (35858, 300, 300, 3)
# * train_chip_coords = 'train_chip_coords.pkl'
#   * an array of bounding boxes per each chipped image
#   * Size: (35858,)
# * train_chip_labels = 'train_chip_labels.pkl'
#   * an array of labels corresponding to the bounding box per each chipped image
#   * Size: (35858,)
# * train_chip_image_names = 'train_chip_image_names.pkl'
#   * an array of original image names per each chipped image
#   * Size: (35858,)

load_from_pkl = False

if (load_from_pkl and os.path.isfile('train_chip_bytes.pkl') and os.path.isfile(
        'train_chip_coords.pkl') and os.path.isfile('train_chip_labels.pkl')):
    train_chip_bytes = torch.load('train_chip_bytes.pkl')
    train_chip_coords = torch.load('train_chip_coords.pkl')
    train_chip_labels = torch.load('train_chip_labels.pkl')
    train_chip_image_names = torch.load('train_chip_image_names.pkl')

else:
    # Here either do preprocessing, or load the pickle files
    train_chip_bytes, train_chip_coords, train_chip_labels, train_chip_image_names = preprocess_dataset(train_path,
                                                                                                        geojson_path,
                                                                                                        chip_size)

# print("train_chip_bytes: ")
# print(np.shape(train_chip_bytes))

# print("train_chip_coords: ")
# print(np.shape(train_chip_coords))

# print("train_chip_labels: ")
# print(np.shape(train_chip_labels))

# print("train_chip_image_names: ")
# print(np.shape(train_chip_image_names))


# -----------------------------------------------------------------------------------
### Save off the outputs so we can share and reuse

# decide if you want to save off the preprocessed data - it is quite time consuming
do_save_pkls = False
if do_save_pkls:
    torch.save(train_chip_bytes, 'train_chip_bytes.pkl')
    torch.save(train_chip_coords, 'train_chip_coords.pkl')  # this seems to take a long time!
    torch.save(train_chip_labels, 'train_chip_labels.pkl')
    torch.save(train_chip_image_names, 'train_chip_image_names.pkl')

# -----------------------------------------------------------------------------------
### Now create the Train Data Loader

def annotation_collate(batch):
    targets = []
    images = []
    pad_size = 0
    longest_target_idx = 0
    for i, item in enumerate(batch):
        images.append(item[0])
        if len(item[1]) > longest_target_idx:
            longest_target_idx = len(item[1])
        targets.append(np.array(item[1]))  # torch.FloatTensor(item[1]))

    for j, t in enumerate(targets):
        amount_to_pad = longest_target_idx - len(t)
        zeros = np.zeros(amount_to_pad)
        targets[j] = np.append(t, zeros)

    return torch.stack(images, 0), np.array(targets)  # torch.stack(images,0) # targets


# transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
# transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print(np.shape(train_chip_bytes))

# training
train_dataset = xViewDataset(image_arr=train_chip_bytes, coord_arr=train_chip_coords, label_arr=train_chip_labels,
                             transform=None)
# train_dataset = DatasetTransformer(train_dataset, transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=annotation_collate)

dtype = torch.float
device = torch.device("cuda:0")

cnn = CNN()
cnn.cuda()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# if cuda:
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.to(device, dtype=torch.float32)).cuda()
        labels = Variable(torch.from_numpy(labels)).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

# -----------------------------------------------------------------------------------
#testing
test_dataset = xViewDataset(image_folder=test_path, groundtruth_filename=geojson_path, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
