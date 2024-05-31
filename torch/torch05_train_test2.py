import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE :', DEVICE)

#1. 데이터 
x = np.array(range(100))
y = np.array(range(1, 101))
print(x)
print(y)
'''
torch :  2.2.0 사용 DEVICE : cuda
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
 96 97 98 99]
[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
  91  92  93  94  95  96  97  98  99 100]
'''
x_train = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)    # reshape(차원 하나 늘리면서 리쉐이프) -> 
y_train = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)    # reshape(차원 하나 늘리면서 리쉐이프) -> 
y_test = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

print(x, y) #tensor([1., 2., 3.]) tensor([1., 2., 3.])

#2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=1))
model = nn.Sequential(
    nn.Linear(1, 5),
    nn.Linear(5, 4),
    nn.ReLU(),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
    nn.Linear(2, 1)
).to(DEVICE)

#3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')
criterion = nn.MSELoss()                #criterion : 표준
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
    return loss2.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 loss : ", loss2)

#result = model.predict([4])
result = model(torch.Tensor([[101]]).to(DEVICE))
print('101의 예측값 : ', result.item())

'''
최종 loss :  0.025720370933413506
101의 예측값 :  101.71670532226562
'''