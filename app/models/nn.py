import torch
import torchvision.transforms as transforms
from torch import nn
from PIL import Image

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

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return torch.argmax(output.data, 1)


if __name__ == '__main__':
    input_size = 784
    hidden_size = 500
    output_size = 10
    model_path = '../../weights/model.pth'
    image_path = '../../uploads/test.png'

    # 加载模型
    model = NN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(model)