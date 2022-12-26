from flask import Flask,request,jsonify

from math import log2
import numpy as np
from scipy.special import expit

class Neural_net:
  def __init__(self,num_layer,list_nodes_per_layer):
    self.num_layer = num_layer
    self.weights = []
    for i in range(num_layer-1):
      self.weights.append(np.random.randn(list_nodes_per_layer[i]+1,list_nodes_per_layer[i+1]) * np.sqrt(1. / list_nodes_per_layer[i+1]))
    self.X = []
    self.Y = []
    self.nue = 0.1

  def sigmoid(self,x):
    return expit(x)

  def dif_sigmoid(self,x):
    ans = self.sigmoid(x)
    return ans * (1-ans)

  def softmax(self,x):
    temp = np.exp(x)
    sum = np.sum(temp,axis=0)
    temp = temp/sum
    return temp

  def dif_softmax(self,x):
    t = self.softmax(x)
    return t*(1-t)
  
  def relu(self,x):
    return (np.maximum(0,x))

  def dif_relu(self,x):
    temp = x.copy()
    temp[temp<=0] = 0
    temp[temp>0] = 1
    return temp

  def one_hot(self,x,categories):
    newY = []
    x = np.reshape(x,(-1))
    size = categories
    for i in range(len(x)):
      temp = np.zeros(size)
      temp[int(x[i])] = 1
      newY.append(temp)
    
    return np.array(newY)
  def set_data(self,d,categories):
    data = np.array(d)
    self.X = np.array(data[:,:-1])
    self.Y = np.array(data[:,-1])
    self.X = self.X.T
    self.Y = self.one_hot(self.Y,categories)
    self.Y = self.Y.T

  def set_weights(self,w):
    for i in range(len(self.weights)):
      self.weights[i] = w[i]

  def get_weights(self):
    return self.weights

  def forward_prop(self,input):
    out_layers = []
    calc_hidden = []
    out_layers.append(input)
    for i in range(self.num_layer-1):
      temp = np.ones(input.shape[1])
      input = np.vstack((temp,input))
      
      calc_hidden.append(self.weights[i].T@input)
      if i==self.num_layer-2:
        input = self.softmax(calc_hidden[i])
      else:
        input = self.sigmoid(calc_hidden[i])
      out_layers.append(input)
    
    ret = [calc_hidden,out_layers]
    return ret

  def back_prop(self,ret,expected):
    zvals = ret[0]
    output = ret[1]
    err_out = expected - output[-1]
    #output layer
    del_out = self.dif_softmax(zvals[-1])*err_out
    #print(del_out)
    del_layers = [None]*(self.num_layer-2)
    #print(del_layers)
    temp = del_out
    for i in range(len(del_layers)-1,-1,-1):
      temp1 = np.vstack((np.ones(zvals[i].shape[1]),self.dif_sigmoid(zvals[i])))
      #print("temp1\n",temp1)
      #print(self.weights[i].shape,temp.shape)
      #print("weight i \n",self.weights[i+1])
      #print("temp\n",temp)
      del_layers[i] = temp1 * (self.weights[i+1]@temp)
      temp = del_layers[i][1:,:]
      del_layers[i] = temp
      #print("del_layer i \n",del_layers[i].shape)
    
    del_layers.append(del_out)
    del_weights = []

    for i in range(self.num_layer-1):
      temp_bias = np.vstack((np.ones(output[i].shape[1]),output[i]))
      #print("del_layer i\n",np.reshape(del_layers[i],(-1)))
      sum = temp_bias @ del_layers[i].T # 785x32 and 32x16
      sum = sum / del_layers[i].shape[1]
      sum = sum * self.nue
      #print("sum\n",sum.shape)
      del_weights.append(sum)

    #print("del_weights\n",del_weights)
    return del_weights

  
  def update_weights(self,epoch,batchsize=4,validation_data=[]):
    history = {'loss':[],'accuracy':[],'val_loss':[],'val_accuracy':[]}
    for _ in range(epoch):
      curX = np.array_split(self.X,int(self.X.shape[1]/batchsize),axis=1) #784x32
      curY = np.array_split(self.Y,int(self.Y.shape[1]/batchsize),axis=1) #10x32
      for i in range(len(curX)):
      #for i in range(1):
        ret = self.forward_prop(curX[i])
        del_weights = self.back_prop(ret,curY[i])

        for j in range(self.num_layer-1):
          self.weights[j] = self.weights[j] + del_weights[j]

      #Calculate Accuracy for epoch
      dataX = self.X.T
      dataY = np.reshape(np.argmax(self.Y,0),(-1,1))
      loss,accuracy = self.evaluate_weights(np.concatenate((dataX,dataY),axis=1))
      history['loss'].append(loss)
      history['accuracy'].append(accuracy)
      if validation_data!=[]:
        val_loss, val_accuracy = self.evaluate_weights(validation_data)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
    return history

  def evaluate_weights(self,testdata):
    xtest = testdata[:,:-1].T
    ytest = testdata[:,-1].T
    # print(xtest)
    ret = self.forward_prop(xtest)
    layer_out = ret[1]

    pred = np.argmax(layer_out[-1],0)
    prob_out = layer_out[-1]
    loss = []
    #print(prob_out)
    for x in range(len(prob_out[0])):
      loss.append(-log2(prob_out[pred[x],x]))
    loss = np.sum(np.array(loss))
    accuracy = np.mean(np.equal(ytest,pred))
    #print("loss, Accuracy : ",loss, accuracy*100)
    return loss , accuracy
  
  def predict(self,x):
    x=np.array(x)
    x=x.T
    result=self.forward_prop(x)
    layer_out=result[1]
    pred = np.argmax(layer_out[-1],0)
    return pred[0]


import pickle

model = pickle.load(open('model (1).pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/predict',methods=['POST'])
def predict():
   bt=float(request.form.get('bt')) 
   bo=float(request.form.get('bo'))
   sh=float(request.form.get('sh'))
   hr=float(request.form.get('hr'))

   input = [[bt,bo,sh,hr]]
   result=float(model.predict(input))
   print(result)
#    result={'bt':bt,'bo':bo,'sh':sh,'hr':hr}
   return jsonify({'r':result})

if __name__ == '__main__':
    app.run(debug=True)