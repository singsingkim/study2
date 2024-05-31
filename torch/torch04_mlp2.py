import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터 
x = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
)

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(x)
print(x.shape, y.shape)  # shape : 구조를 보여줌 (2, 10) (10,)
x = x.T
print(x.shape)  # (10, 2)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

print(x, y)

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1)
).to(DEVICE)

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()                #criterion : 표준
# optimizer = optim.Adam(model.parameters(), lr = 0.01)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# model.fit(x,y, epochs = 100, batch_size=1)
def train(model, criterion, optimizer, x, y):
    # model.train()   #훈련모드 default
    
    optimizer.zero_grad()
    # w = w - lr * (loss를 weight로 미분한 값)
    hypothesis = model(x) #예상치 값 (순전파)
    loss = criterion(hypothesis, y) #예상값과 실제값 loss
    
    #역전파
    loss.backward() #기울기(gradient) 계산 (loss를 weight로 미분한 값)
    optimizer.step() # 가중치(w) 수정(weight 갱신)
    return loss.item() #item 하면 numpy 데이터로 나옴

epochs = 1000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch {}, loss: {}'.format(epoch, loss)) #verbose

print("===================================")

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model, criterion, x, y):
    model.eval() #평가모드

    with torch.no_grad():
        y_predict = model(x)
        loss2 = criterion(y, y_predict)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print("최종 loss : ", loss2)

#result = model.predict([4])
result = model(torch.Tensor([[10,1.3,0]]).to(DEVICE))
print('4의 예측값 : ', result.item())