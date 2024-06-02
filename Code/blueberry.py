from keras.src import optimizers
from keras.src.layers import Dense
from keras.src.layers import Input
from keras.src import Model
from keras.src.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from path import bpath

def plot_loss_curve(history):
    plt.figure(figsize = (5,4))
    
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc ='upper right')
    
    plt.show()

data = pd.read_csv(bpath + "Data/WildBlueberryPollinationSimulationData.csv")

idata = data.iloc[:, 0:13]
odata = data.iloc[:, 16]

train_idata = idata.sample(frac=0.8,random_state=0)
train_odata = odata.sample(frac=0.8,random_state=0)
test_idata = idata.drop(train_idata.index)
test_odata = odata.drop(train_odata.index)


m = Sequential()
input = Input((13,))
h1 = Dense(100, activation="sigmoid")(input)
h2 = Dense(50, activation="relu")(h1)
h3 = Dense(25, activation="relu")(h2)
h4 = Dense(12, activation="relu")(h3)
output = Dense(1)(h4)
m = Model(input, output)

m.summary()
m.compile(optimizers.Adam(learning_rate=1e-6), loss='mae')
history=m.fit(train_idata, train_odata, epochs=1200000, validation_split = 0.3, batch_size=30)
m.save(bpath+'Model/blueberry.keras')
# plot_loss_curve(history)