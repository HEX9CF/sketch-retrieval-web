import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def predict(self, image):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(image)
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x)
])

if __name__ == '__main__':
    input_size = 784
    hidden_size = 500
    output_size = 10
    model_path = '../../weights/model.pth'
    image_path = '../../temp/img.png'

    # 加载模型
    model = NN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(model)

    # 加载图片
    image = Image.open(image_path)
    # plt.imshow(image)
    # plt.show()

    image = transform(image)
    image = image.view(1, -1)
    print(image.shape)

    # 预测
    predicted = model.predict(image)
    print(predicted)