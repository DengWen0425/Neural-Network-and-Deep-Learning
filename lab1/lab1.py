import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import linear_model
from sklearn.svm import SVC
import pickle
import os
from torchvision import transforms
from PIL import Image
import time
import re


def sum_all_numbers(l):
    """
    sum all numbers in a list
    :param l: input list
    :return: a number
    """
    if type(l) != list:
        raise TypeError("The input parameter type must be a list!")
    result = 0
    for ele in l:
        if type(ele) in [int, float]:
            result += ele
    return result


def list2unique(l):
    """
    Given a list, return its all unique elements
    :param l: input list
    :return: list with unique elements
    """
    if type(l) != list:
        raise TypeError("The input parameter type must be a list!")
    result = []
    for ele in l:
        if ele not in result:
            result.append(ele)
    return result


def is_palindrom(s):
    """
    judge a string s is palindrom or not
    :param s: input string, can be word, phrase or sequence
    :return: a bool factor
    """
    s = s.lower()  # In this problem, the algorithm is not case sensitive.
    i = 0
    j = len(s) - 1
    while i < j:
        if s[i] == " ":
            i += 1
            continue
        if s[j] == " ":
            j -= 1
            continue
        if s[i] != s[j]:
            return False
        i += 1
        j -= 1
    return True


def complex_split(x):
    """
    ﬁnd the real and imaginary parts of an array of complex numbers
    :param x: Input array x, contains some complex numbers
    :return:array of real and imaginary parts of every complex numbers
    """
    result = []
    for ele in x:
        result.append(np.array([ele.real, ele.imag]))
    return np.array(result)


def binary_add(x, y):
    """
    to add two binary numbers
    :param x: a string like "1001"
    :param y: a string like "1"
    :return: a string like "1010"
    """
    if x.count("1") + x.count("0") != len(x) or y.count("1") + y.count("0") != len(y):
        raise ValueError("The input string must only contain 1 and 0")
    result = ""
    i, j = len(x) - 1, len(y) - 1
    car = 0
    while i >= 0 and j >= 0:
        tmp = int(x[i]) + int(y[j]) + car
        cur = tmp % 2
        car = tmp // 2
        result = str(int(cur)) + result
        i -= 1
        j -= 1
    while i >= 0:
        tmp = int(x[i]) + car
        cur = tmp % 2
        car = tmp // 2
        if car + cur > 0 or j != 0:
            result = str(int(cur)) + result
        i -= 1
    while j >= 0:
        tmp = int(y[j]) + car
        cur = tmp % 2
        car = tmp // 2
        if car + cur > 0 or j != 0:
            result = str(int(cur)) + result
        j -= 1
    if car:
        result = str(int(car)) + result
    return result


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def list_node_add(x, y):
    """
     Add the two numbers represented by listNode and return it as a linked list.
    :param x: a listNode storing a number
    :param y: a listNode storing a number
    :return:a listNode storing number = x + y
    """
    result = None
    p = None
    car = 0
    while x and y:
        tmp = x.val + y.val + car
        cur = tmp % 10
        if result is None:
            result = ListNode(cur)
            p = result
        else:
            p.next = ListNode(cur)
            p = p.next
        car = tmp // 10
        x = x.next
        y = y.next
    while x:
        tmp = x.val + car
        cur = tmp % 10
        p.next = ListNode(cur)
        p = p.next
        car = tmp // 10
        x = x.next
    while y:
        tmp = y.val + car
        cur = tmp % 10
        p.next = ListNode(cur)
        p = p.next
        car = tmp // 10
        y = y.next
    if car:
        p.next = ListNode(car)
    return result


def bubble_sort(nums):
    """
    sort the list of numbers using bubble sort.
    :param nums: the input list of numbers
    :return: list of numbers already sorted
    """
    for i in range(len(nums) - 1):
        for j in range(len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums


def merge_sort(nums):
    """
    sort the list of numbers using merge sort
    :param nums: the input list of numbers
    :return: list of numbers already sorted
    """
    if len(nums) <= 1:
        return nums
    pivot = int(len(nums) / 2)
    left = merge_sort(nums[:pivot])
    right = merge_sort(nums[pivot:])
    return merge(left, right)


def merge(left, right):
    """
    merge two sorted list together
    :param left: the left list
    :param right:  the right list
    :return: a sorted list
    """
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += list(left[i:])
    result += list(right[j:])
    return result


def quick_sort(nums):
    """
    sort the list of numbers using quick sort
    :param nums: the input list of numbers
    :return: list of numbers already sorted
    """
    if len(nums) <= 1:
        return nums
    pivot = nums[int(len(nums) / 2)]
    left, right = [], []
    for i in range(len(nums)):
        if i == int(len(nums) / 2):
            continue
        elif nums[i] >= pivot:
            right.append(nums[i])
        else:
            left.append(nums[i])
    return quick_sort(left) + [pivot] + quick_sort(right)


def shell_sort(nums):
    """
    sort the list of numbers using shell's sort
    :param nums: the input list of numbers
    :return: list of numbers already sorted
    """
    gap = len(nums) // 2
    while gap > 0:
        for i in range(gap, len(nums)):
            temp = nums[i]
            j = i
            while j >= gap and nums[j - gap] > temp:
                nums[j] = nums[j - gap]
                j -= gap
            nums[j] = temp
        gap = gap // 2
    return nums


class LinearRegressionModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.linear(x)
        return out


"""# input data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
y = np.array([2.2, 3.8, 6.4, 9, 9.5, 12, 13.8, 17, 18, 21], dtype=np.float32)
x = x.reshape((10, 1))
y = y.reshape((10, 1))


# initialize hyper parameters, model,  loss function and optimizer
learning_rate = 0.01
epochs = 50
in_size = 1
out_size = 1
model = LinearRegressionModel(in_size, out_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

x_train = Variable(torch.from_numpy(x).float())
y_train = Variable(torch.from_numpy(y).float())

for epoch in range(epochs):

    optimizer.zero_grad()  # empty the gradients
    outputs = model(x_train)  # forward

    loss = criterion(outputs, y_train)  # compute the loss
    loss.backward()  # backward

    optimizer.step()  # update parameters

    if (epoch + 1) % 10 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, epochs, loss.item()))

model.eval()
pred = model(x_train)
pred = pred.detach().numpy()
plt.plot(x, y, "bx", label="Original Data")
plt.plot(x, pred, label="Predictions")
plt.legend()
plt.show()

# torch.save(model.state_dict())"""


class LogisticRegressionModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.linear(x))
        return out


"""# initialize hyper parameters, model,  loss function and optimizer
learning_rate = 0.01
epochs = 50
in_size = 2
out_size = 1
model = LogisticRegressionModel(in_size, out_size)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# input data
x = np.array([[1, 9], [2, 11], [5, 1], [8, 0], [4, 9.5], [6, 2.5],
              [2.2, 10.3], [7.5, 1.9], [9.2, 0.7], [0.88, 6.2]], dtype=np.float32)
y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=np.float32)

y = y.reshape((10, 1))
# autograd
x_train = Variable(torch.from_numpy(x).float())
y_train = Variable(torch.from_numpy(y).float())

for epoch in range(epochs):

    optimizer.zero_grad()  # empty the gradients
    outputs = model(x_train)  # forward

    loss = criterion(outputs, y_train)  # compute the loss
    loss.backward()  # backward

    optimizer.step()  # update parameters

    if (epoch + 1) % 10 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, epochs, loss.item()))

model.eval()
pred = model(x_train)
pred = pred.detach().numpy()
print("Predict on train set")
print(pred)
print("---convert: if > 0.5: 1 else: 0---")
print(np.int64(pred > 0.5))
print("-----Test-----")
print("Input: (0.5, 9.2):", sep="")
print(model(torch.from_numpy(np.array([[0.5, 9.2]])).float()).detach().numpy()[0])
print("Input: (6.7, 1.2):", sep="")
print(model(torch.from_numpy(np.array([[6.7, 1.2]])).float()).detach().numpy()[0])

# torch.save(model.state_dict())"""


class SupportVectorMachine(nn.Module):
    def __init__(self, in_size):
        super(SupportVectorMachine, self).__init__()
        self.linear = nn.Linear(in_size, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


def hing_loss(outputs, labels):
    loss = torch.clamp(1 - outputs * labels, min=0)
    loss = torch.mean(loss)
    return loss


def hing_loss_LF1(outputs, labels, w, alpha=0.01):
    loss = torch.clamp(1 - outputs * labels, min=0)
    loss = torch.mean(loss)
    loss += alpha * torch.norm(w, p="fro")
    return loss


def hing_loss_LF2(outputs, labels, w, alpha=0.01):
    loss = torch.clamp(1 - outputs * labels, min=0)
    loss = torch.mean(loss)
    loss += alpha * (w ** 2).sum()
    return loss


"""# initialize hyper parameters, model,  loss function and optimizer
learning_rate = 0.05
epochs = 500
in_size = 2
out_size = 1
model = SupportVectorMachine(in_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# input data
x, y = make_blobs(n_samples=100, centers=2, random_state=3, cluster_std=[1, 1])  # use sklearn to generate data sets
# Plot the data
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
# feature scaling
x = (x - x.mean())/x.std()
# replace 0 with -1
y[y == 0] = -1
y = np.reshape(y, (100, 1))

# autograd
x_train = Variable(torch.from_numpy(x).float())
y_train = Variable(torch.from_numpy(y).float())

for epoch in range(epochs):

    optimizer.zero_grad()  # empty the gradients
    outputs = model(x_train)  # forward

    loss = hing_loss(outputs, y_train)  # compute the loss
    loss.backward()  # backward

    optimizer.step()  # update parameters

    if (epoch + 1) % 10 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'.format(epoch + 1, epochs, loss.item()))

# Visualize
W = model.linear.weight[0].data.cpu().numpy()
b = model.linear.bias[0].data.cpu().numpy()

delta = 0.01  # sep of coordinates
# coordinates
_x = np.arange(x[:, 0].min(), x[:, 0].max(), delta)
_y = np.arange(x[:, 1].min(), x[:, 1].max(), delta)
_x, _y = np.meshgrid(_x, _y)
coordinates = list(map(np.ravel, [_x, _y]))

# draw the different areas
z = (W.dot(coordinates) + b).reshape(_x.shape)
z[np.where(z > 1.)] = 4
z[np.where((z > 0.) & (z <= 1.))] = 3
z[np.where((z > -1.) & (z <= 0.))] = 2
z[np.where(z <= -1.)] = 1

# draw the picture
plt.xlim([x[:, 0].min() + delta, x[:, 0].max() - delta])
plt.ylim([x[:, 1].min() + delta, x[:, 1].max() - delta])
plt.contourf(_x, _y, z, alpha=0.8, cmap="Greys")
plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y))
plt.tight_layout()
plt.show()"""

"""# generate data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
x = x.reshape(-1, 1)
y = np.array([2.2, 3.8, 6.4, 9, 9.5, 12, 13.8, 17, 18, 21], dtype=np.float32)

# Model import
lm = linear_model.LinearRegression()

# Fit a linear regression
lm.fit(x, y)

# learned parameters
print("Coefficient:", lm.coef_)
print("Intercept:", lm.intercept_)

# visualize
pred = lm.predict(x)

plt.plot(x, y, "bx", label="Original Data")
plt.plot(x, pred, label="Predictions")
plt.legend()
plt.show()"""

"""# generate data
x = np.array([[1, 9], [2, 11], [5, 1], [8, 0], [4, 9.5], [6, 2.5],
              [2.2, 10.3], [7.5, 1.9], [9.2, 0.7], [0.88, 6.2]], dtype=np.float32)
y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=np.float32)

# model import
lm = linear_model.LogisticRegression(solver="liblinear")

# fit a regression
lm.fit(x, y)

# learned parameters
print("Coefficient:", lm.coef_)
print("Intercept:", lm.intercept_)

# predict on train set
pred = lm.predict(x)
print("Predict on train set")
print(pred)

# predict on new data
print("-----Test-----")
print("Input: (0.5, 9.2):", sep="")
print(lm.predict([[0.5, 9.2]]))
print("Input: (6.7, 1.2):", sep="")
print(lm.predict([[6.7, 1.2]]))"""

"""# generate data
x, y = make_blobs(n_samples=100, centers=2, random_state=3, cluster_std=[1, 1])  # use sklearn to generate data sets
# Plot the data
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
# feature scaling
x = (x - x.mean())/x.std()
# replace 0 with -1
y[y == 0] = -1

# model import
model = SVC(kernel="linear")

# fit
model.fit(x, y)

# Visualize

delta = 0.01  # sep of coordinates
# coordinates
_x = np.arange(x[:, 0].min(), x[:, 0].max(), delta)
_y = np.arange(x[:, 1].min(), x[:, 1].max(), delta)
_x, _y = np.meshgrid(_x, _y)
coordinates = np.array(list(map(np.ravel, [_x, _y])))
coordinates = coordinates.transpose()

# draw the different areas
z = model.decision_function(coordinates)
z[np.where(z > 1.)] = 4
z[np.where((z > 0.) & (z <= 1.))] = 3
z[np.where((z > -1.) & (z <= 0.))] = 2
z[np.where(z <= -1.)] = 1
z = z.reshape(_x.shape)

# draw the picture
plt.xlim([x[:, 0].min() + delta, x[:, 0].max() - delta])
plt.ylim([x[:, 1].min() + delta, x[:, 1].max() - delta])
plt.contourf(_x, _y, z, alpha=0.8, cmap="Greys")
plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y))
plt.tight_layout()
plt.show()
"""


# load single batch
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# load whole data
def load_cifar():
    """
    load all train and test data in cifar10
    :return: train,  test
    """
    labels = []
    data = []

    for b in range(1, 6):
        f = os.path.join('./cifar-10-batches-py/data_batch_%d' % (b,))
        d = unpickle(f)
        labels.append(d[b'labels'])
        data.append(d[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int"))
    train_labels = np.concatenate(labels)
    train_data = np.concatenate(data)
    d = unpickle('./cifar-10-batches-py/test_batch')
    labels = d[b'labels']
    data = d[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("int")
    return train_data, train_labels, data, labels


"""# all labels
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# load data
train_data, train_labels, test_data, test_labels = load_cifar()
# visualize some images
choices = np.random.choice(50000, 4)
for i in range(4):
    choice = choices[i]
    plt.subplot(1, 4, i+1)
    plt.imshow(train_data[choice])
    plt.title(classes[train_labels[choice]])
plt.show()"""

# all labels
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class DatasetCIFAR10(Dataset):
    def __init__(self, root='./cifar-10-batches-py/', train=True, transform=None, target_transform=None):
        self.labels = []
        self.data = []
        self.transform = transform
        self.target_transform = target_transform

        # the way of loading data is similar to the function defined in problem 16
        if train:  # to decide which dataset to load: train set or test set
            for b in range(1, 6):
                f = os.path.join(root, 'data_batch_%d' % (b,))
                d = unpickle(f)
                self.labels.append(d[b'labels'])
                self.data.append(d[b'data'].reshape(10000, 3, 32, 32).astype("int"))
            self.labels = np.concatenate(self.labels)
            self.data = np.concatenate(self.data)
        else:
            d = unpickle(root + 'test_batch')
            self.labels = d[b'labels']
            self.data = d[b'data'].reshape(10000, 3, 32, 32).astype("int")

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # to support transform operation
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.data)


"""train_loader = DataLoader(DatasetCIFAR10(), 4, shuffle=False)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

import torchvision
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))"""

# class torchvision.transforms.Scale(size, interpolation=2)
"""
Change the size of the input 'PIL. Image' to the given 'size'. The 'size' is the minimum side length. 
For example, if the `height>width` of the original image, 
the size of the image after changing the size is' (size*height/width, size) '.
"""

"""# --------Case-------------
crop = transforms.CenterCrop(300)
img = Image.open('test.jpg')

plt.subplot(121)
plt.imshow(img)
print(img.size)
croped_img = crop(img)
plt.subplot(122)
plt.imshow(croped_img)
print(croped_img.size)
plt.show()"""

# class torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
"""
Randomly change the brightness, contrast and saturation of an image.
"""

"""# --------Case-------------
transform = transforms.ColorJitter((0, 0.5))

img = Image.open('test.jpg')

plt.subplot(121)
plt.imshow(img)
tran_img = transform(img)
plt.subplot(122)
plt.imshow(tran_img)
plt.show()"""

# class torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None, fill=0)
"""
Rotate the image by angle.
"""
"""# --------Case-------------
transform = transforms.RandomRotation((90, 90))

img = Image.open('test.jpg')

plt.subplot(121)
plt.imshow(img)
tran_img = transform(img)
plt.subplot(122)
plt.imshow(tran_img)
plt.show()"""

# class torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
"""
Pad the given PIL Image on all sides with the given “pad” value.
"""
"""# --------Case-------------
transform = transforms.Pad(50)

img = Image.open('test.jpg')

plt.subplot(121)
plt.imshow(img)
tran_img = transform(img)
plt.subplot(122)
plt.imshow(tran_img)
plt.show()"""

"""transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='.', train=True, transform=transform)


batch_sizes = [1, 4, 64, 1024]
num_workers = [0, 1, 4, 16]
pin_memory = [False, True]
iters = []
for batch_size in batch_sizes:
    for num_worker in num_workers:
        for pin_bool in pin_memory:
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=num_worker,
                                                       pin_memory=pin_bool)
            num = 0
            start = time.time()
            for idx, data in enumerate(train_loader):
                num += 1
            end = time.time()
            iters.append(num)
            print(
                "batch_size={}\tnum_workers={}\tpin_memory={}\truntime:{:.2f}s".format(batch_size, num_worker, pin_bool,
                                                                                       end - start))"""

"""transform = transforms.Compose(
    [transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='.', train=True, transform=transform)
data = trainset.data

means = []
stds = []
for i in range(3):
    channel_img = data[:, :, :, i]/255
    means.append(channel_img.mean())
    stds.append(channel_img.std())

print("mean \t R:{} \t G:{} \t b:{}".format(means[0], means[1], means[2]))
print("std \t R:{} \t G:{} \t b:{}".format(stds[0], stds[1], stds[2]))"""

characters = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")


def img2char(img, char_list, width=100, height=100, save=True, out_file=None):
    """
    To convert an image to character painting
    :param img:  Input image
    :param char_list: Corresponding characters
    :param width: The output width
    :param height: The output height
    :param save:  Whether to save to .txt file
    :param out_file: Output file name
    :return:  character painting
    """
    length = len(char_list)
    distance = 256 / length
    new_img = img.resize((width, height), Image.ANTIALIAS)
    painting = ""
    for i in range(width):
        for j in range(height):
            pixel = new_img.getpixel((i, j))
            pixel = pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722
            painting += char_list[int(pixel / distance)]
        painting += "\n"

    if save:
        if out_file is None:
            out_file = 'character_painting'
        with open(out_file + ".txt", 'w') as f:
            f.write(painting)

    return painting


"""img = Image.open("test2.jpg")
paint = img2char(img, characters)
print(paint)"""

# -----------Question 1----------------
"""data = np.random.random((10, 2))  # Randomly generated a 10x2 matrix
x, y = data[:, 0], data[:, 1]  # separate the coordinates
r = np.sqrt(x**2 + y**2).reshape(-1, 1)  # compute r
theta = np.arctan(y/x).reshape(-1, 1)  # compute theta
polar = np.concatenate([r, theta], axis=1)  # gather together

# print the cartesian coordinates and polar coordinates.
print("------cartesian coordinates-------")
print(data)
print("------  polar   coordinates-------")
print(polar)"""


# -----------Question 2----------------


class Symmetric2DArray(np.ndarray):
    # overwrite a np.ndarray class

    def __setitem__(self, index, value):  # make the array is symmetric
        i, j = index
        super(Symmetric2DArray, self).__setitem__((i, j), value)
        super(Symmetric2DArray, self).__setitem__((j, i), value)


def symmetrize(z):
    """
    convert z to symmetric matrix
    :param z: input 2d array
    :return:  symmetric 2d array
    """
    return np.asarray(z + z.T - np.diag(z.diagonal())).view(Symmetric2DArray)


"""# an example
S = symmetrize(np.random.randint(0, 10, (4, 4)))
S[1, 3] = 999
print(S)"""


# -----------Question 3----------------


def get_distance(P0, P1, P):
    P10 = P1 - P0
    P10_mod = (P10 ** 2).sum(axis=1)
    all_distance = []
    for p in P:
        dot = ((p - P0) * P10).sum(axis=1)
        tmp = (dot / P10_mod).reshape(-1, 1) * P10
        dist = P0 - p + tmp
        all_distance.append(np.sqrt((dist ** 2).sum(axis=1)))
    return np.array(all_distance)


"""P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))

y = get_distance(P0, P1, p)

print("x")"""


def bilinear_interp(A, point):
    """
    Bilinear Interpolation Algorithm.
    :param A: Given points
    :param point: Given point which need to compute
    :return: the result of Interpolation
    """
    x, y = point
    x -= 1
    y -= 1
    try:
        return A[x][y]
    except IndexError:
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1
        result = A[x1][y1] * (x2 - x) * (y2 - y) + A[x2][y1] * (x - x1) * (y2 - y) + A[x1][y2] * (x2 - x) * (y - y1) + \
                 A[x2][y2] * (x - x1) * (y - y1)
        return int(result)


"""# test
A = np.array([[110, 120, 130], [210, 220, 230], [310, 320, 330]])
print(bilinear_interp(A, (1, 1)))
print(bilinear_interp(A, (2.5, 2.5)))"""


def cartesian_product(sets):
    """
    compute cartesian product of the set
    :param sets: iterable set
    :return: the cartesian product
    """
    tmp_set = [[x] for x in sets[0]]
    for i in range(1, len(sets)):
        result = []
        for tmp in tmp_set:
            for ele in sets[i]:
                result.append(tmp + [ele])
        tmp_set = result
    return tmp_set


"""example = [[1, 2, 3], [4, 5], [6, 7]]
print(cartesian_product(example))"""


def subset(Z, pos, shape, fill=0):
    """
    extract a subset form an array
    :param Z: input array
    :param shape: shape of subset
    :param fill: padding element
    :param pos: subset position
    :return: subset
    """
    result = np.ones(shape)*fill
    x, y = pos
    x_len, y_len = shape
    rx, ry = x_len//2, y_len//2
    x1 = max(x-rx, 0)
    y1 = max(y-ry, 0)
    x2 = min(x + rx, Z.shape[0])
    y2 = min(y + ry, Z.shape[1])
    for i in range(x1, x2):
        for j in range(y1, y2):
            result[i+rx-x][j+ry-y] = Z[i][j]
    return result


"""# example:
Z = np.random.randint(0, 10, (5, 5))
print(Z)
pos = (1, 1)
shape = (4, 4)
fill = 0
print(subset(Z, pos, shape, fill))"""


# ----------sub-problem.1----------
def add(A, B):
    """
    add operation of matrix
    :param A:  matrix A
    :param B:  matrix B
    :return:  result
    """
    return [[A[i][j] + B[i][j] for j in range(len(A[i]))] for i in range(len(A))]


"""if len(A) != len(B):
        raise ValueError("The two input array must have the same dimensions")
    result = []
    for i in range(len(A)):
        if len(A[i]) != len(B[i]):
            raise ValueError("The two input array must have the same dimensions")
        tmp = []
        for j in range(len(A[i])):
            tmp.append(A[i][j] + B[i][j])
        result.append(tmp)
    return result
"""


# ----------sub-problem.2----------
def subtract(A, B):
    """
    subtract operation of matrix
    :param A:  matrix A
    :param B:  matrix B
    :return:  result
    """
    return [[A[i][j] - B[i][j] for j in range(len(A[i]))] for i in range(len(A))]


# ----------sub-problem.3----------
def scalar_multiply(A, num):
    """
    scalar mulitply operation of matrix
    :param A:  matrix A
    :param num:  number of scalar
    :return:  result
    """
    return [[A[i][j] * num for j in range(len(A[i]))] for i in range(len(A))]


# ----------sub-problem.4----------
def multiply(A, B):
    """
    multiply operation of matrix
    :param A:  matrix A
    :param B:  matrix B
    :return:  result
    """
    result = []
    for i in range(len(A)):
        tmp = []
        for j in range(len(B[0])):
            tmp.append(sum([A[i][k] * B[k][j] for k in range(len(A[0]))]))
        result.append(tmp)
    return result


# ----------sub-problem.5----------
def identity(num):
    """
    create a identity mat
    :param num:  the dim of the mat
    :return:  the identity matrix
    """
    return [[1 if i == j else 0 for j in range(num)] for i in range(num)]


# ----------sub-problem.6----------
def transpose(A):
    """
    transpose a mat
    :param A: input mat
    :return: mat after transposed
    """
    result = []
    for i in range(len(A[0])):
        tmp = []
        for j in range(len(A)):
            tmp.append(A[j][i])
        result.append(tmp)
    return result


# ----------sub-problem.7----------
def get_minor(A, i, j):
    """
    compute the minor of the mat at pos(i,j)
    :param A: input mat
    :param i: pos[0]
    :param j: pos[1]
    :return: minor
    """
    return [row[:j] + row[j+1:] for row in (A[:i]+A[i+1:])]


def get_det(A):
    """
    compute the det of a mat
    :param A: the input mat A
    :return:  the det of A
    """
    # base case for 2x2 matrix
    if len(A) == 2:
        return A[0][0]*A[1][1]-A[0][1]*A[1][0]

    det = 0
    for c in range(len(A)):
        det += ((-1)**c)*A[0][c]*get_det(get_minor(A, 0, c))
    return det


def inverse(A):
    """
    compute the inverse of a mat
    :param A: input mat
    :return: mat after inverse
    """
    det = get_det(A)
    # special case for 2x2 matrix:
    if len(A) == 2:
        return [[A[1][1]/det, -1*A[0][1]/det],
                [-1*A[1][0]/det, A[0][0]/det]]

    # find matrix of cofactors
    result = []
    for r in range(len(A)):
        cofactorsRow = []
        for c in range(len(A)):
            minor = get_minor(A, r, c)
            cofactorsRow.append(((-1)**(r+c)) * get_det(minor))
        result.append(cofactorsRow)
    result = transpose(result)
    for r in range(len(result)):
        for c in range(len(result)):
            result[r][c] = result[r][c]/det
    return result


"""# ---------Test examples-----------
matrix_a = [[12, 10], [3, 9]]
matrix_b = [[3, 4], [7, 4]]
matrix_c = [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]]
matrix_d = [[3, 0, 2], [2, 0, -2], [0, 1, 1]]

print("------------add----------")
print(add(matrix_a, matrix_b))
print("-------subtract----------")
print(subtract(matrix_a, matrix_b))
print("-----scalar_multiply-----")
print(scalar_multiply(matrix_b, 3))
print("---------multiply--------")
print(multiply(matrix_a, matrix_b))
print("--------identity---------")
print(identity(3))
print("---------transpose-------")
print(transpose(matrix_c))
print("---------inverse---------")
print(inverse(matrix_d))"""


def GCD(x, y):
    """
    find the greatest common divisor using Euclidean Algorithm
    :param x: num
    :param y: num
    :return: the gcd
    """
    tmp = x % y
    while tmp != 0:
        x = y
        y = tmp
        tmp = x % y
    return y


"""# ----------Examples----------
print("GCD(3, 5):", GCD(3, 5))
print("GCD(6, 3):", GCD(6, 3))
print("GCD(-2, 6):", GCD(-2, 6))
print("GCD(0, 3):", GCD(0, 3))"""


def consecutive_seq(N):
    """
    Find all consecutive positive number sequences whose sum is N
    :param N: input num
    :return: all sequence
    """
    if N == 0 or N == 1:
        return []
    result = []
    i = 1
    tmp = [i]
    while i <= N//2+1 and i+1 != N:
        i += 1
        tmp = tmp + [i]
        cur_sum = sum(tmp)
        while cur_sum > N:
            tmp = tmp[1:]
            cur_sum = sum(tmp)
        if cur_sum == N:
            result.append(tmp.copy())
    return result


"""# ----------Example----------
result = consecutive_seq(2)
for ele in result:
    print(ele)"""


def password_checking(s):
    """
    to check whether a sequence of input strings are qualified
    :param s: input sequence of strings
    :return: sequence of valid string
    """
    result = []

    strings = s.split(",")
    for password in strings:
        # at least a letter between [a-z]
        if not re.search("[a-z]", password):
            continue
        # At least 1 number between [0-9]
        if not re.search("[0-9]", password):
            continue
        # At least 1 letter between [A-Z]
        if not re.search("[A-Z]", password):
            continue
        # At least 1 character from [$#@]
        if not re.search("[$#@]", password):
            continue
        # Minimum length of transaction password: 6
        if len(password) < 6:
            continue
        if len(password) > 12:
            continue
        result.append(password)

    return ",".join(result)


passwords = "ABd1234@1,a F1#,2w3E*,2We3345"
print(password_checking(passwords))

