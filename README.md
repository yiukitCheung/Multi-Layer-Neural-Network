# Neural Network Numpy Implementation 
## Createing a Neural Network from Scratch
Create different layers classes to form a multi-layer nerual network with various type of regularization method and optimization method.

## Build status
[NumPy] https://numpy.org/doc/stable/index.html
[Matplotlib] https://matplotlib.org/stable/index.html

## Features
Include an Adam optimizer for gradient descent computation.
Include an early stop function to save unnecessary training time

## How to run the code

Run the code block sequentially. Most importantly, run everything before Part 3. Train a Multi-layer Neural Network

## How to create and customize a neural network

1. Create a model() class
2. Append layers to model() accordingly
3. Set the loss function, optimization method, and evaluation metric for the model()
4. Finalize model() to connect hidden layers accordingly
5. Train model() using model.train()
6. Predict using model(). predict

## How to make prediction

1. confidences = model().predict(X_test)
2. Predictions = model().output_layer_activation.predictions(confidences)


## Examples
Ex 1. To train a 3-layer neural network with sigmoid activation in 50 epochs :

model = Model()
model.add(Layer_Dense(n_input,n_output))
model.add(Activation_sigmoid())
model.add(Layer_Dense(n_input,n_output)
model.add(Activation_sigmoid())
model.add(Layer_Dense(n_input,n_output)
model.add(Activation_softmax())
model.set(
	loss = Loss_Cross_Entropy(), 
	optimizer = SGD(learning_rate=0.03),
	 accuracy = Accuracy_Cate()
)
model.train(X,y,epochs=50,batch_size=None,print_by=128,verbose=-1,threshold=0.001)

Ex 2. To create a normalized mini-batch 3-layer neural network using relu activation along with Adam Optimization and weight decay in 50 epoch:

model = Model()
model.add(Layer_Dense(n_input,n_output, norm = True, weight_reg_l2=0.01))
model.add(Activation_ReLu())
model.add(Layer_Dense(n_input,n_output, norm = True, weight_reg_l2=0.01))
model.add(Activation_ReLu())
model.add(Layer_Dense(n_input,n_output, norm = True, weight_reg_l2=0.01))
model.add(Activation_softmax())
model.set(
	loss = Loss_Cross_Entropy(), 
	optimizer = Adam(learning_rate=0.03, decay=1e-3,epsilon=1e-7,beta_1=0.9,beta_2=				0.999),
	 accuracy = Accuracy_Cate()
)

model.train(X,y,epochs=50,batch_size=128,print_by=128,verbose=-1,threshold=0.001)

threshold : The minimum accuracy gain for the model to continue training
verbose : set to -1 is print the training summary




