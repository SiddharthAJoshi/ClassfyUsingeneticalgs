from geneticalgs import BinaryGA, RealGA, DiffusionGA, MigrationGA
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.metrics import mean_squared_error 
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math

seed=9
np.random.seed(seed)
#input the cyrotherapy dataset

tran=np.loadtxt("Cyrotherapy.csv",delimiter=",",skiprows=1)
x=tran[:,0:6]
y=tran[:,6]
# target = tran[7:7]
# print target

#splitting into training and testing data sets
(x_train,x_test,y_train,y_test)= train_test_split(x,y,test_size=0.25,random_state=seed)


model = Sequential()
model.add(Dense(12, input_dim=6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
t0 = time.time()
#compliation of the model
history= model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=1000, batch_size=512)

t1 = time.time()

print ("Time taken to train the model: ", t1-t0, "secs")
# accuracy 
scores=model.evaluate(x_test,y_test)
print("Accuracy: %.2f%%" %(scores[1]*100))
 # code refernece take from from https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
print(history.history.keys())

plt.plot(history.history['loss'])

plt.title("Back Propogation ---Accuracy: %.2f%%" %(scores[1]*100))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# from tpot import TPOTRegressor

# tpot = TPOTRegressor(generations=5, population_size=90, verbosity=2)
# tpot.fit(x_train, y_train)

# tpot.export('tpot




# reference taken from github geneticalgs examples
# GA standard settings
generation_num = 50
population_size = 30
elitism = True
selection = 'rank'
tournament_size = None # in case of tournament selection
mut_type = 1
mut_prob = 0.05
cross_type = 1
cross_prob = 0.95
optim = 'min' # minimize or maximize a fitness value? May be 'min' or 'max'.
interval = (-1,1)

# Migration GA settings
period = 5
migrant_num = 3
cloning = True

def func(x):
    return abs(x*(math.sin(x/11)/5 + math.sin(x/110)))

x1 = list(range(1000))
y1 = [func(elem) for elem in x1]
a = np.array(y1[0:72]).reshape(6,12)
b = np.array(y1[72:84])
c = np.array(y1[84:96]).reshape(12,1)
d = np.array(y1[96:97])

model.layers[0].set_weights([a,b])
model.layers[1].set_weights([c,d])

weights = model.layers[0].get_weights()[0]
biases = model.layers[0].get_weights()[1]
print("Wieght ")
print(weights)
print("biases")
print(biases)

def errorfunc (x):

	a = np.array(y1[0:72]).reshape(6,12)
	b = np.array(y1[72:84])
	c = np.array(y1[84:96]).reshape(12,1)
	d = np.array(y1[96:97])
	model.layers[0].set_weights([a,b])
	model.layers[1].set_weights([c,d])
   
	y_predict = model.predict(x_train)
	y_predict=y_predict.flatten()
 
	root_mean_squared_error = np.sum(np.square(y_predict - y_train))
	return root_mean_squared_error

sga = RealGA(errorfunc, optim=optim, elitism=elitism, selection=selection,
            mut_type=mut_type, mut_prob=mut_prob, 
            cross_type=cross_type, cross_prob=cross_prob)

print("Wait There is Still more analysis")
sga.init_random_population(population_size, 91, interval)
fitness_progress = sga.run(generation_num)

print("--------------------------------------------------------------------------")
sga.best_solution
best_weight = sga.best_solution[0]
print("\n\n")
print("BEST Wieghts")
print(best_weight)
y_test_predict = model.predict(x_test)

print(y_test)

print("Predicted Values")
print(y_test_predict)

confusion_matrix = confusion_matrix(y_test, y_test_predict)
print("Confusion Matrix for GA ")
print("\n")
print(confusion_matrix)

accuracy = accuracy_score(y_test,y_test_predict)
print('Accuracy Score: ', accuracy, '\n')






