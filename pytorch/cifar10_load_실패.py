import torch
from cifar10_save import 모델, device, test_data

# 모델 호출
path = 'C:\Study\pytorch\_save\\'
model = 모델().to(device)
model.load_state_dict(torch.load(path + 'cifar10_model.pth'))
print('cifar10_model.pth_모델 호출 완료')


# 4 예측
classes = [
    
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print('예측됌 : ', predicted, '\n실제 : ', actual)