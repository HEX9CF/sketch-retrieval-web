import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import nn
from app.models.vgg16 import vgg16

SKETCH_VGG = '../../data/model/sketch_vgg16_5.pth'

PHOTO_ROOT = '../../data/'

photo_data = pickle.load(open('../../data/feature/photo-vgg16-5.pkl', 'rb'))
photo_feature = photo_data['feature']
photo_name = photo_data['name']

nbrs = NearestNeighbors(n_neighbors=np.size(photo_feature, 0),
                        algorithm='brute', metric='euclidean').fit(photo_feature)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载模型
def load_model():
    model = vgg16(pretrained=False)
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
def retrieve_images(feature):
    retrieval_images = []
    feature = np.reshape(feature, [1, np.shape(feature)[0]])
    distances, indices = nbrs.kneighbors(feature)

    for i, idx in enumerate(indices[0][:5]):
        retrieval_name = photo_name[idx]
        print(i, retrieval_name)
        retrieval_image = Image.open(os.path.join(PHOTO_ROOT, retrieval_name)).convert('RGB')
        retrieval_images.append(retrieval_image)

    return retrieval_images

# 展示检索结果
def show_retrieval(sketch, photos):
    plt.subplot(1, 6, 1)
    plt.title('sketch')
    plt.imshow(sketch)

    for i, photo in enumerate(photos):
        plt.subplot(1, 6, i + 2)
        plt.imshow(photo)
        plt.title(f'photo{i + 1}')

    plt.show()

if __name__ == '__main__':
    image_path = 'test.png'

    # 加载图片
    test_image = Image.open(image_path)

    # 图片预处理
    test_image = transform(test_image)
    test_image = test_image.unsqueeze(0)

    model = load_model()
    feature = extract_feature(model, test_image)
    retrievals = retrieve_images(feature)

    test_image = Image.open(image_path)
    show_retrieval(test_image, retrievals)









