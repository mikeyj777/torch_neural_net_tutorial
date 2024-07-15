import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# parameters
rand_seed = 41
lr = 0.01
epochs = 100

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
        x = F.relu(self.fx2(x))
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train) # long tensor = 64-bit int
y_test = torch.LongTensor(y_test)

# set criterion to measure error, how far off predictions are from 
criterion = nn.CrossEntropyLoss()

# choose Adam optimizer, set learning rate

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

losses = []

for _ in range(epochs):
    # get predicted results
    y_pred = model.forward(X,X_train)

    # measure loss
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    



apple = 1