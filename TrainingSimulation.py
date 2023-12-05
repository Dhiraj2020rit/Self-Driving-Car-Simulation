import matplotlib.pyplot as plt
print('Setting Up, Please hold up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from  sklearn.model_selection import  train_test_split

# step 1 prepare data
path = 'DataDirectory'
data = importDataInfo(path)

# step 2 visualization
data = balanceData(data, display=False)

# step 3
imagesPath, steerings = loadData(path, data)
# print(imagesPath[0], steering[0])

# step 4 Splitting Data
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
# print('Total Training Images: ', len(xTrain))
# print('Total Validation Images: ', len(xVal))

# step 6 preprocessing

# step 7

# step 8
model = createModel()
model.summary()

# Step 9 train

history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10, validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)

# step 10 save and plot data

model.save('model.h5')
print('Model Saved Successfully')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('losses')
plt.xlabel('Epoch')
plt.show()
