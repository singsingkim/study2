# 07. load_diabetes
# 08. california
# 09. dacon ddarung
# 10. kaggle bike

# 평가는 rmse, r2
#######################################################
# (logistic_regression = 이진 분류)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape)  # (442, 10)
print(y.shape)  # (442,)



x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=333,
    # stratify=y,
)

# x_train = torch.FloatTensor(x_train).to(DEVICE)    # reshape(차원 하나 늘리면서 리쉐이프)
# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# x_test = torch.FloatTensor(x_test).to(DEVICE)    # reshape(차원 하나 늘리면서 리쉐이프)
# y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
### 여기까지의 데이터형식은 넘파이. 아래 과정에서 텐서로 변경

x_train = torch.FloatTensor(x_train).to('cpu')    # reshape(차원 하나 늘리면서 리쉐이프)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to('cpu')
x_test = torch.FloatTensor(x_test).to('cpu')    # reshape(차원 하나 늘리면서 리쉐이프)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to('cpu')

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 8),
    nn.Linear(8, 1),
    # nn.Softmax()
).to('cpu')

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()                #criterion : 표준
# criterion = nn.BCELoss()                #criterion : 표준, 이진분류
optimizer = optim.Adam(model.parameters(), lr = 0.01)
# optimizer = optim.SGD(model.parameters(), lr = 0.0001)

# model.fit(x,y, epochs = 100, batch_size=1)
# 배치단위로 돌아간다
def train(model, criterion, optimizer, x_train, y_train):
    # model.train()   #훈련모드 default
    
    optimizer.zero_grad()   # -> loss를 weight로 편미분한 값을 초기화 시킨다는 것 (0 으로)
    # w = w - lr * (loss를 weight로 편미분한 값)
    hypothesis = model(x_train) #예상치 값 (순전파)
    loss = criterion(hypothesis, y_train) #예상값과 실제값 loss
    
    #역전파
    loss.backward() #기울기(gradient) 계산 (loss를 weight로 미분한 값)  # 역전파 시작
    optimizer.step() # 가중치(w) 수정(weight 갱신)                     # 역전파 끝
    return loss.item() #item 하면 numpy 데이터로 나옴

epochs = 20000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch {}, loss: {}'.format(epoch, loss)) #verbose

print("===================================")

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model, criterion, x_test, y_test):
    model.eval() #평가모드  # 훈련할때는 드롭아웃 적용, 노말라이션 적용하지만
                            # 평가할때는 훈련가중치로 1에포만 돌리면서 모든 데이터를
                            # 평가하기때문에 드롭아웃 해제, 노말라이션 해제 시킨다
                            # model.eval() 을 적용하지 않으면 
                            # 기본적으로 model.train 을 적용시킨다

    with torch.no_grad():   # 토치는 자동으로 그라디언트 건든다. 그라디언트 건들지 않는다고 설정
                            # 역전파 사용안함으로 설정 -> 이걸 하지 않으면 평가하면서 그라디언트 갱신된다
                            
        y_predict = model(x_test)
        loss2 = criterion(y_test, y_predict)
    return loss2.item(), y_predict

loss2, y_predict = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)

############################# 밑에 어떻게 될까 ##################################

#result = model.predict([4])
# result = model(torch.Tensor([[101]]).to(DEVICE))
# acc = accuracy_score(y_test, y_predict.round())
r2 = r2_score(y_test, y_predict)
# r2 = r2_score(y_test, y_predict.round())
# print('101의 예측값 : ', result.item())
# print('acc : ', acc)
print('r2 : ', r2)

'''
최종 loss :  12199.294921875
f1 :  -1.3302563659608087
'''
