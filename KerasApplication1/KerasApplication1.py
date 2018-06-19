#Importy
from keras.models import Sequential
from keras.layers import Dense
import numpy

#with open("losowe-liczby.txt") as f:
#    lines=f.readlines()
#    textfile=open("output.txt","w")
#    for i in lines:
#        textfile.write(i[0:-2]+"\n")
#    textfile.close()

# fix random seed for reproducibility
numpy.random.seed(6)
X=numpy.empty([1000,15])
Y=numpy.empty([1000,5])
x_test=numpy.empty([2,15])
# load weather dataset
dataset = numpy.loadtxt("output.txt", delimiter=",")
# split into input (X) and output (Y) variables
dataset2=dataset[:,[1,4,7,10,16]]

for i in range(997):
    X[i]=numpy.append(dataset2[i,:],[dataset2[i+5,:],dataset2[i+10,:]])
    #print(X[i])
    Y[i]=dataset2[i+11,:]
print(X[1])
x_test[0]=numpy.append(dataset2[1000,:],[dataset2[1001,:],dataset2[1002,:]])
x_test[1]=numpy.append(dataset2[1000,:],[dataset2[1001,:],dataset2[1002,:]])
print(x_test)
# create model
model = Sequential()
model.add(Dense(16, input_dim=15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='relu'))

print("Składanie modelu")
# Compile model
model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['mse'])
#Test czy działa
print("Model złożony teraz uczenie")
# Fit the model/Uczenie
model.fit(X, Y, epochs=120, batch_size=20)
#model.fit(X, Y, epochs=30, batch_size=5)

# evaluate the model
scores = model.evaluate(X, Y)
#print(numpy.transpose(x_test))
classes = model.predict(x_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(classes)
input()