import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import geojson
import json
import time

def chip_image1(img, chip_size=(300, 300)):
    """
    Segment an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    width, height, _ = img.shape
    wn, hn = chip_size
    images = np.zeros((int(width / wn) * int(height / hn), wn, hn, 3))
    k = 0
    for i in tqdm(range(int(width / wn))):
        for j in range(int(height / hn)):
            chip = img[wn * i:wn * (i + 1), hn * j:hn * (j + 1), :3]
            images[k] = chip

            k = k + 1

    return images.astype(np.uint8)

with open(fname) as f:
        data = json.load(f)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, conv_size, kernel_size=kernel_size, padding=2),
            nn.BatchNorm2d(conv_size),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv_size, conv_size*2, kernel_size=kernel_size, padding=2),
            nn.BatchNorm2d(conv_size*2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(conv_size * in_channels * (conv_size*2), num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
# -----------------------------------------------------------------------------------
cnn = CNN()
cnn.cuda()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
#if cuda:
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
    for images, labels in train_loader:
        np.shape(images)
        np.shape(lables)
        images = torchvision.transforms.functional.to_tensor(images)
        np.shape(images)
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')