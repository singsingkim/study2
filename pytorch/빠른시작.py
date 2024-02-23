import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 1. 데이터
# 공개 데이터셋에서 학습 데이터 내려받음
training_data = datasets.FashionMNIST(
    root='C:\_data\data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# 공개 테스틑셋에서 학습 데이터 내려받음
test_data = datasets.FashionMNIST(
    root='C:\_data\data',
    train=False,
    download=True,
    transform=ToTensor(),
)


BATCH_SIZE = 64

# 데이터로더를 생성
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for x, y in test_dataloader:
    print(f'Shape of x [N, C, H, W] : {x.shape}')
    print(f'Shape of y : {y.shape}{y.dtype}')
    break


# 2. 모델 생성
# 학습에 사용할 cpu, gpu, mps(multi process service) 장치를 얻음
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)
print(f'Using {device} device')

# 모델 정의
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.aaa = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.aaa(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# 모델 매개변수 최적화
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 트레인
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x,y) in enumerate(dataloader):
        x,y = x.to(device), y.to(device)
        
        # 예측 오류 계산
        pred = model(x)
        loss = loss_fn(pred,y)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(x)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
            
# 테스트
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy : {(100*correct):>0.1f}%, Avg loss : {test_loss:>8f}\n')
    
# 학습
epochs = 5
for t in range(epochs):
    print(f'epoch {t+1}\n--------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print('Done!!!!!!!!!!!')    


# # 모델 저장
path = 'C:\Study\pytorch\_save\\'
torch.save(model.state_dict(), path + '빠른시작_model.pth')
print('Saved PyTorch Model State to model.pth')

# 모델 호출
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