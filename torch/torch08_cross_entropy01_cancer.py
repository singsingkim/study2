import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42, shuffle=True, stratify=y)

scaler =StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape, y_train.shape) 

#2. 모델구성
model = nn.Sequential(
    nn.Linear(30, 64),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 8),
    nn.Linear(8, 1),
    nn.Sigmoid()
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()               #criterion : 표준
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(model, criterion, optimizer, x, y):
    model.train()  # 훈련 모드
    optimizer.zero_grad()
    hypothesis = model(x)  # 예상치 값 (순전파)
    loss = criterion(hypothesis, y)  # 예상값과 실제값의 loss
    loss.backward()  # 기울기(gradient) 계산 (loss를 weight로 미분한 값)
    optimizer.step()  # 가중치(w) 수정 (weight 갱신)
    return loss.item()  # item 하면 numpy 데이터로 나옴

epochs = 3000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch {}, loss: {}'.format(epoch, loss)) #verbose

print("===================================")

#4. 평가, 예측
def evaluate(model, criterion, x, y):
    model.eval() #평가모드
    x, y = x.to(DEVICE), y.to(DEVICE)

    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)

results = model(x_test)
results = np.round(results.cpu().detach().numpy())
print(results)

# y_test를 CPU로 이동하여 numpy 배열로 변환
y_test_cpu = y_test.cpu().numpy().squeeze()
print('예측값 : ', results)
acc = accuracy_score(results, y_test_cpu)
print("acc : ", acc)