import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler



# GPU 를 되는지 안되는지 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ' , torch.__version__ , '사용 DEVICE : ', DEVICE)


#1. 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

print(x,y)  # tensor([1., 2., 3.]) tensor([1., 2., 3.])
print(x.shape,y.shape)

# x = torch.FloatTensor(x)                     # 이걸 해줘야지 torch data로 사용할 수 있음 // unsqueeze(1) 2 차원을 맞춰주는 것
# y = torch.FloatTensor(y).unsqueeze(1)        # shape를 똑같이 해줘야 된다 // 똑같지 않으면 그냥 n 빵 때려서 중간값으로 1개가 들어가게 됨 이 때는 2가 들어가게 됨    

print(x,y)  # tensor([1., 2., 3.]) tensor([1., 2., 3.])
print(x.shape,y.shape) 

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2 , shuffle = True , random_state = 3 , stratify = y)

print(x_train.shape,y_train.shape)      # (569, 30) (569,)
print(x_test.shape,y_test.shape)        # (455, 30) (455,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE).unsqueeze(1)
y_test = torch.FloatTensor(y_test).to(DEVICE).unsqueeze(1)

print(x_train.shape, x_test.shape)      # (114, 30) (114,)

# x와y를 합친다
from torch.utils.data import TensorDataset

train_set  = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# print(type(train_set))        # <torch.utils.data.dataset.TensorDataset object at 0x7f2f53721190>
# print(print(train_set.shape))   # AttributeError: 'TensorDataset' object has no attribute 'shape'
print(len(train_set))       # 455


#토치 데이터 만들기, 배치 넣어준다~!! 끝!! 

from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(len(train_loader))    # 15

######################################### 데이터 끝 #############################################






## 2 모델구성
# model = nn.Sequential(
#     nn.Linear(30,64),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.Linear(16,7),
#     nn.Linear(7,1),
#     nn.Sigmoid()
#     ).cuda() # 인풋 , 아웃풋  # y = xw + b

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__()
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 19)
        self.linear2 = nn.Linear(19, 97)
        self.linear3 = nn.Linear(97, 9)
        self.linear4 = nn.Linear(9, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        
        return
    
    # 순전파!!
    def forward(self, input_size):      
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        
        return x
    
    
# model = Model(input_dim, output_dim)    
model = Model(30, 1).to(DEVICE)


    
    
    
#3 컴파일
# model.compile(loss = 'mse' , optimizer = 'adam' )
# criterion = nn.MSELoss()                  # criterion : 표준 -> 이렇게 쓰는 이유는 그냥 제일 많이 써서
criterion = nn.BCELoss()                    # binary_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters() , lr=0.01 )         # SGD = Batch 안에 mini batch를 만들어서 그 안에 있는 것 중 랜덤으로 1개를 지정해서 훈련하고 나머지는 버림
                                                            # 연산량이 빨라지고 성능이 좋을 수 도 있음 , 다른 에포에서는 다른 것을 쓸 수 있어서 데이터를 100% 버린다고는 할 수 없음

# model.fit(x,y,epochs = 100 , batch_size = 1)
def train(model , criterion , optimizer, loader):
    
    total_loss = 0
    
    for x_batch, y_batch in loader:
        
    
    
        ####### 순전파 #######
        # model.train()     # 훈련모드 , 디폴트 값 // 훈련모드랑 dropout , batch normalization 같은것을 사용
        # w = w - lr * (loss를 weight로 미분한 값)
        optimizer.zero_grad()       # zero_grad = optimizer를 0으로 초기화 시킨다
                                    # 1. 처음에 0으로 시작하는게 좋아서
                                    # 2. epoch가 돌때마다 전의 gradient를 가지고 있어서 그게 문제가 될 수 있어서 이걸 해결 하기 위해서
                                    #    계속 0으로 바꿔주는 것이다. 

        hypothesis = model(x_batch)       # 예상치 값 (순전파)
        
        loss = criterion(hypothesis , y_batch)    #예상값과 실제값 loss
        
        #####################
        
        loss.backward()         # 기울기(gradient)값(loss를 weight로 미분한 값) 계산 -> 역전파 시작
        optimizer.step()        # 가중치 수정(w 갱신)       -> 역전파 끝
        # total_loss = total_loss + loss.item()
        total_loss += loss.item()        
        
        return total_loss / len(loader)      # total_loss / 13
    
    
    
epochs = 1000
for epoch in range(1 , epochs+ 1 ) :
    loss = train(model , criterion, optimizer , train_loader )
    print('epoch : {} , loss:{}'.format(epoch,loss))    # verbose

print('==========================================================')

#4. 평가, 예측
# loss = model.evaluate(x,y)
def evaluate(model , criterion , loader ) : 
    model.eval()            # 평가모드 , 안해주면 평가가 안됨 dropout 같은 것들이 들어가게 됨
    
    total_loss = 0
    
    for x_batch, y_batch in loader:
    
        with torch.no_grad():
            y_predict = model(x_batch)
            loss2 = criterion(y_batch , y_predict)
            total_loss += loss2.item()
            
        return total_loss / len(loader)

loss2 = evaluate(model , criterion , test_loader)
print('최종 loss : ', loss2 )

# result = model.predict([4])
result = model(x_test)
print('예측값은 :' , result.tolist() )

from sklearn.metrics import accuracy_score

# with torch.no_grad():
#     y_pred = model(x_test).cpu().numpy().squeeze()

# y_test = y_test.cpu().numpy().squeeze() 

y_pred = model(x_test).cpu().detach().numpy()   # detach 는 gradient를 안받는 것 // Tensor 데이터 뒤에 기울기가 붙는데 그걸 없애줘야 메모리를 효율적으로 쓰고 영향을 받지 않는다.

y_pred = np.around(y_pred) 
y_test = y_test.cpu().numpy()

acc = accuracy_score(y_test,y_pred)
print('acc :{:.4f} '.format(acc))

# 최종 loss :  0.00011766229727072641
# 4의 예측값은 : 4.021754264831543

