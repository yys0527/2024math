from keras.src import optimizers
from keras.src.layers import Dense
from keras.src.layers import Input
from keras.src import Model
from keras.src.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
from path import bpath                          # 보안을 위한 경로 은닉 모듈 (직접 구동 시 삭제)

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


m = Sequential()
input = Input((13,))
h1 = Dense(1000, activation="relu")(input)
h2 = Dense(1000, activation="relu")(h1)
h3 = Dense(1000, activation="relu")(h2)
h4 = Dense(1000, activation="relu")(h3)
output = Dense(1)(h4)
m = Model(input, output)

# m.summary()
m.compile(optimizers.Adam(learning_rate=1e-4), loss='mae')
history=m.fit(idata, odata, epochs=12000, validation_split = 0.3)
m.save(bpath+'Model/blueberry1e-4.keras')
# plot_loss_curve(history)
