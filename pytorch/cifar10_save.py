import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 1 데이터
# 트레인 데이터 내려받기
train_data = datasets.CIFAR10(
    root='C:\_data\data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# 테스트 데이터 내려받기
test_data = datasets.CIFAR10(
    root='C:\_data\data',
    train=False,
    download=True,
    transform=ToTensor(),
)

   
BATCH_SIZE = 256
# 배치사이즈 256
#  정확도 : 40.5%, 평균 로스 : 1.6622591257095336)


# 데이터로더 생성
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for x, y in test_dataloader:
    print(f'x 쉐이프 : {x.shape}')  
    # x 쉐이프 : torch.Size([32, 3, 32, 32])
    print(f'y 쉐이프 : {y.shape}')
    # y 쉐이프 : torch.Size([32])
    break
    
# 2 모델 생성
# cpu, gpu, mps 사용여부
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)    
    
print(f'사용중인 장치 : {device}')
# 사용중인 장치 : cuda

'''
# DNN 모델 정의
class  모델(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )        

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits
'''

# CNN 모델 정의
class 모델(nn.Module):
    def __init__(self):
        super(모델, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128*4*4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x





model = 모델().to(device)
print(model)

# 모델 매개변수 최적화
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 트레인
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 전체 데이터의 갯수
    for batch, (x,y) in enumerate(dataloader):
    # (dataloader)를 통해 제공되는 각 미니배치에 대해 반복하면서
    # 각 미니배치의 인덱스와 해당 미니배치의 입력 
    # 데이터(x)와 정답(y)을 순차적으로 가져오는 코드

        x,y = x.to(device), y.to(device)
                
        # 예측 오류 계산
        predict = model(x)
        loss = loss_fn(predict, y)
        
        # 역전파
        optimizer.zero_grad()
        # 객체가 관리하는 모든 모델 파라미터의 변화도(gradient)를 0으로 초기화
        loss.backward()
        # 손실 함수(loss function)에 대한 역전파(backpropagation)를 수행
        optimizer.step()
        # 모델의 파라미터를 업데이트
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(x)
            # (batch+1) * len(x)는 현재까지 처리된 총 데이터의 수를 계산
            print(f'loss : {loss:7f} [{current:>5d}/{size:>5d}]')
            # loss 값과 데이터 처리가 어느정도까지 되었는지 확인
            
# 테스트
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 전체 데이터의 갯수            
    num_batches = len(dataloader)   # 미니배치 단위로 데이터셋을 처리할 때의 미니배치 수
    model.eval()
    # 모델을 평가(evaluation) 모드로 설정. 모델을 평가 모드로 설정하면, 
    # 학습 중에 일어나는 변화를 방지하고, 예측만을 수행합니다. 
    # 즉, 드롭아웃(dropout)이나 배치 정규화(batch normalization)와 같은 학습 시에만
    # 적용되는 연산들이 평가 시에는 적용되지 않습니다.
    test_loss, correct = 0,0
    with torch.no_grad():
        for x, y in dataloader:
            x, y  = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
            # (pred.argmax(1) == y)는 모델의 예측값과 실제 정답값이 같은지를 
            # 나타내는 불리언(boolean) 텐서를 생성합니다. 
            # 예측이 맞으면 True를, 틀리면 False를 갖습니다.
            # type(forch.float) 는 불리언 텐서를 부동소수점텐서로 변환
            # 부동소수점 텐서들의 모든 합
            
    test_loss /= num_batches
    correct /= size
    print(f'테스트 에러 : \n 정확도 : {(100*correct):>0.1f}%, 평균 로스 : {test_loss:>8})\n')


# 학습
epochs = 5
for t in range(epochs):
    print(f'epoch {t+1}\n' + '-'*50)
    train(train_dataloader, model, loss, optimizer)
    test(test_dataloader, model, loss)
print('완료!!!!!!!!!!!!!!!!!!!!!!!')


# 모델 저장
path = 'C:\Study\pytorch\_save\\'
torch.save(model.state_dict(), path + 'cifar10_model.pth')
print('cifar10_model.pth_모델저장')



# 4 예측
classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print('예측됌 : ', predicted, '\n실제 : ', actual)


# epoch 5
# --------------------------------------------------
# loss : 0.812365 [  256/50000]
# loss : 0.850005 [25856/50000]
# 테스트 에러 : 
#  정확도 : 68.6%, 평균 로스 : 0.9101470381021499)

# 완료!!!!!!!!!!!!!!!!!!!!!!!
# cifar10_model.pth_모델저장
# 예측됌 :  cat 
# 실제 :  cat