import pickle

import numpy as np
import torch
import torch.nn as nn
import math
from PIL import Image
from matplotlib import pyplot as plt

from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import transforms

from app.config import *


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print('feature.size()', x.size())
        feature = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

photo_data = pickle.load(open(PHOTO_FEATURE, 'rb'))
photo_feature = photo_data['feature']
photo_name = photo_data['name']

nbrs = NearestNeighbors(n_neighbors=np.size(photo_feature, 0),
                        algorithm='brute', metric='euclidean').fit(photo_feature)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载模型
def load_model():
    model = VGG(make_layers(cfg['D']))
    model.classifier[6] = nn.Linear(in_features=4096, out_features=125, bias=True)
    model.load_state_dict(torch.load(SKETCH_VGG, map_location=torch.device('cpu'), weights_only=True), strict=False)
    return model

# 提取特征
def extract_feature(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)

    feature = output[1].squeeze().numpy()

    return feature

# 检索图片
def retrieve_images(feature, num):
    retrieval_images = []
    feature = np.reshape(feature, [1, np.shape(feature)[0]])
    distances, indices = nbrs.kneighbors(feature)

    for i, idx in enumerate(indices[0][:num]):
        retrieval_name = photo_name[idx]
        print(i, retrieval_name)
        retrieval_image = Image.open(os.path.join(DATA_ROOT, retrieval_name)).convert('RGB')
        retrieval_images.append(retrieval_image)

    return retrieval_images

# 展示检索结果
def show_retrieval(sketch, photos):
    num = len(photos)
    plt.subplot(1, num + 1, 1)
    plt.title('sketch')
    plt.imshow(sketch)

    for i, photo in enumerate(photos):
        plt.subplot(1, num + 1, i + 2)
        plt.imshow(photo)
        plt.title(f'photo{i + 1}')

    plt.show()

if __name__ == '__main__':
    test_image_path = '../../data/test.png'

    # 加载图片
    test_image = Image.open(test_image_path)

    # 图片预处理
    test_image = transform(test_image)
    test_image = test_image.unsqueeze(0)

    model = load_model()
    feature = extract_feature(model, test_image)
    retrievals = retrieve_images(feature, 5)

    test_image = Image.open(test_image_path)
    show_retrieval(test_image, retrievals)









