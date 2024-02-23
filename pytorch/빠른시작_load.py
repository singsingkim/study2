import torch
from 빠른시작_save import NeuralNetwork, device, test_data


# 모델 호출
path = 'C:\Study\pytorch\_save\\'
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(path + '빠른시작_model.pth'))


# 4 예측
classes = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot',
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted : "{predicted}", Actual : "{actual}"')