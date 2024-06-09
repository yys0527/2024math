from keras._tf_keras.keras.models import load_model
from path import bpath
import pandas as pd
import matplotlib.pyplot as plt

m = load_model(bpath+"Model/le-4_1000-1000-1000-1000_12000/model.keras")

data = pd.read_csv(bpath + "Data/WildBlueberryPollinationSimulationData.csv")

idata = data.iloc[:, 0:13]
odata = data.iloc[:, 16]

test_predictions = m.predict(idata)
a = 0.0

for i in range(len(odata)):                                             # 표준편차 구현부
    a += abs(odata[i]-test_predictions[i])
print(a/len(odata))                                                     

plt.scatter(odata, test_predictions.flatten(), alpha = 0.5, s = 3)      # 산점도 구현부
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([0, 10000], [0, 10000], color='black')
plt.show()