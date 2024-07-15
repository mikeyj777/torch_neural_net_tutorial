import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# parameters
rand_seed = 41
lr = 0.01
test_size = 0.2
epochs = 100
output_every_x_epochs = epochs // 10


# create model class that inherits nn.Module
class Model(nn.Module):
    # input layer (4 features of flower) ->
    # hidden layer (h1) -> 
    # hidden layer (h2) ->
    # output (3 classes of iris flowers)
    
    # features - sepal length & width, pedal len & width

    def __init__(self, in_features=4, h1 = 8, h2 = 9, out_features = 3) -> None:
        # in_features = 4 - 4 flower features (sepal length & width, pedal len & width)
        # h1 = 8, h2 = 9 - num neurons in each hidden layer
        # out_features - images classified into one of 3 classes
        
        super().__init__() # instantiate super class (nn.Module)
        self.fc1 = nn.Linear(in_features=in_features, out_features=h1) # fc1 = "fully connected layer 1".  connects from input to h1
        self.fc2 = nn.Linear(in_features=h1, out_features=h2)
        self.out = nn.Linear(in_features=h2, out_features=out_features)
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x)) # relu - value above zero.  zero otherwise
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# set your seed
torch.manual_seed(rand_seed)

model = Model()

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)

# change last column.  str -> float
dict_map = {
    'Setosa': 0,
    'Versicolor': 1,
    'Virginica': 2,
}

my_df['variety'] = my_df['variety'].map(dict_map)

# Train Test Split!

# drop the outcome col.  isolate input features.
# get numpy vals
X = my_df.drop('variety', axis=1)
X = X.values

y = my_df['variety'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_seed)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train) # long tensor = 64-bit int
y_test = torch.LongTensor(y_test)

# set criterion to measure error, how far off predictions are from 
criterion = nn.CrossEntropyLoss()

# choose Adam optimizer, set learning rate

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []

for i in range(epochs):
    # get predicted results
    y_pred = model.forward(X_train)

    # measure loss
    loss = criterion(y_pred, y_train)
    loss_np = loss.detach().numpy()
    loss_val = loss_np
    losses.append(loss_val)

    if i % output_every_x_epochs == 0:
        print(f'epoch: {i}. loss: {loss}')

    # back prop - take error rate in forward prop, feed back thru network to fine tune weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.plot(range(epochs), losses)
plt.ylabel('error')
plt.xlabel('Epoch')
# plt.show()

# evaluate model on test data set
with torch.no_grad(): # turn off back prop
    y_eval = model.forward(X_test) #X_test - features from test set.  y_eval is the preds from it
    loss = criterion(y_eval, y_test)
    print(f'test loss: {loss}')

correct= 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        y_val_item = y_val.argmax().item()
        #what type of flower our network thinks it is
        check = y_val_item == y_test[i]
        print(f'iter: {i} | {str(y_val)} | y_val vs y_test: {y_val_item} vs {y_test[i]} good? {check}')
        if check:
            correct += 1
    
    print(f'correct {correct}')
        

# add new data
# with torch.no_grad():
#     new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
#     print(model(new_iris)) # predicts "0 - Setosa"

#     # should be a 2 - "Virginica"
#     newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])
#     print(model(newer_iris)) # predicts 2

# save trained network model
torch.save(model.state_dict(), 'data/torch_nn_iris_1_model.pt')

new_model = Model()
new_model.load_state_dict(torch.load('data/torch_nn_iris_1_model.pt'))

print(new_model.eval())

apple = 1