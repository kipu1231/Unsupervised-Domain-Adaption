import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Function


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamb

        return output, None

def grad_reverse(x, lamb):
    return GradientReverse.apply(x, lamb)


class Dann(nn.Module):
    def __init__(self, args):
        super(Dann, self).__init__()

        self.conv_model = torch.nn.Sequential(
            # 3x28x28 --> 32x26x26
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxPool2d(2),
            #nn.Dropout2d(0.25, True),

            # 32x26x26 --> 64x12x12
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25, True),

            # 64x12x12 --> 128x5x5
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3, True),

        )

        self.label_class = torch.nn.Sequential(
            nn.Linear(3200, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 10)
        )

        self.domain_class = torch.nn.Sequential(
            nn.Linear(3200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 2)
        )

    def forward(self, img, lamb):
        x = self.conv_model(img)
        # nach dem Conv hat der Tensor die Größe (batchsize, channels, x, y)
        # um das ganze dem Classifier zu übergeben, reshaped man den Tensor und
        # macht exakt batchsize columns und entsprechend viele rows
        x = x.view(x.size(0), -1)
        reversal_layer = grad_reverse(x, lamb)
        img_label = self.label_class(x)
        domain_label = self.domain_class(reversal_layer)

        return img_label, domain_label


class DannSource(nn.Module):
    def __init__(self, args):
        super(DannSource, self).__init__()

        self.conv_model = torch.nn.Sequential(
            # 3x28x28 --> 32x26x26
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            #nn.MaxPool2d(2),
            #nn.Dropout2d(0.25, True),

            # 32x26x26 --> 64x12x12
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25, True),

            # 64x12x12 --> 128x5x5
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3, True),

        )

        self.label_class = torch.nn.Sequential(
            nn.Linear(3200, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),

            nn.Linear(100, 10)
        )

    def forward(self, img):
        x = self.conv_model(img)
        # nach dem Conv hat der Tensor die Größe (batchsize, channels, x, y)
        # um das ganze dem Classifier zu übergeben, reshaped man den Tensor und
        # macht exakt batchsize columns und entsprechend viele rows
        x = x.view(x.size(0), -1)
        img_label = self.label_class(x)

        return img_label


class Adda_net(nn.Module):
    def __init__(self, args):
        super(Adda_net, self).__init__()

        self.conv = torch.nn.Sequential(
            # 3x28x28 --> 32x26x26
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.MaxPool2d(2),
            #nn.Dropout2d(0.25, True),

            # 32x26x26 --> 64x12x12
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25, True),

            # 64x12x12 --> 128x5x5
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3, True),

            # # 3x28x28 --> 32x13x13
            # nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1),
            # nn.ReLU(True),
            # nn.MaxPool2d(2),
            # # nn.MaxPool2d(2),
            # nn.Dropout2d(0.25, True),
            #
            # # 64x13x13 --> 64x12x12
            # nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            # nn.MaxPool2d(2),
            # nn.Dropout2d(0.3, True),
        )

    def forward(self, img):
        x = self.conv(img)
        return x


class Adda_classifier(nn.Module):
    def __init__(self, args):
        super(Adda_classifier, self).__init__()

        self.classifier = torch.nn.Sequential(
            nn.Linear(3200, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 10)
        )

    def forward(self, img):
        x = img.view(img.size(0), -1)
        x = self.classifier(x)
        return x


class Adda_discriminator_mnistm_svhn(nn.Module):
    def __init__(self, args):
        super(Adda_discriminator, self).__init__()

        self.discriminator = torch.nn.Sequential(
            nn.Linear(3200, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 2)
        )

    def forward(self, img):
        x = img.view(img.size(0), -1)
        x = self.discriminator(x)
        return x


class Adda_discriminator(nn.Module):
    def __init__(self, args):
        super(Adda_discriminator, self).__init__()

        self.discriminator = torch.nn.Sequential(
            nn.Linear(3200, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 2)
        )

    def forward(self, img):
        x = img.view(img.size(0), -1)
        x = self.discriminator(x)
        return x

