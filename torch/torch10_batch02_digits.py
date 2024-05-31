import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (1797, 64) (1797,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42, shuffle=True, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)  # Long -> 정수형 CrossEntropy에 들어가면 원핫을 실행해줌
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, y_train.shape)  # (1257, 64) (1257,)

# torch 데이터 만들기 1. x와 y를 합침
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# torch 데이터 만들기 2. 배치 넣어줌.
print(len(train_set))  # 1257
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# 2. 모델 구성
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

model = Model(64, 10).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()  # criterion : 표준
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()  # batch 당 초기화
        hypothesis = model(x_batch)  # 예상치 값 (순전파)
        loss = criterion(hypothesis, y_batch)  # 예상값과 실제값의 loss
        loss.backward()  # 기울기(gradient) 계산 (loss를 weight로 미분한 값)
        optimizer.step()  # 가중치(w) 수정 (weight 갱신)
        total_loss += loss.item()
    return total_loss / len(loader)  # item 하면 numpy 데이터로 나옴

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch % 100 == 0:
        print('epoch {}, loss: {}'.format(epoch, loss))  # verbose

print("===================================")

# 4. 평가, 예측
def evaluate(model, criterion, loader):
    model.eval()  # 평가모드
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

loss2 = evaluate(model, criterion, test_loader)
print("최종 loss : ", loss2)

# 예측값 계산 및 정확도 평가
model.eval()

total_acc = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_predict = model(x_batch)
        y_predict = torch.argmax(y_predict, dim=1)
        # y_predict와 y_batch를 CPU로 이동하여 numpy 배열로 변환
        y_predict = y_predict.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        total_acc += accuracy_score(y_batch, y_predict)
print("acc : ", total_acc / len(test_loader))