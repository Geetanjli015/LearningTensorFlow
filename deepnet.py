import tensorflow as tf
#MNSIT handwrittern recognition

# ...
# input > weight > hidden layer 1 (activation function) > weight > hidden layer 2 
# (activation function) > weights > output layer
# ...

#feed forward neural network (passing data straight through)
# compare output to intended output > cost function (cross entropy) (how close is it)
# optimization function -> To minimize cost (AdamOptimizer ... SGD,AdaGrad)
# backward motion and manipulation of weights is called backpropagation
# one cycle -> feed forward + backpropagation = epoch

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

#10 classes that is handwrittern digits from 0 to 9
#one hot mean only one is on rest are off
#0 = [1,0,0,0,0,0,0,0,0,0,0]
#1 = [0,1,0,0,0,0,0,0,0,0,0]
#Example so on...only one pixel or one element is on

#we have three hidden layer -->i.e deep neural network
#number of nodes of each neural network..these doesn't have to identical..it depends on number of layers and model
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#number of classes is 10 for mnist ..though we can derive
n_classes = 10

batch_size =100
#it will go through batches of 100 images and feed through a network at a time 
#and manipulate the weights 

#28*28 flaten out to 784 pixels or values .. as long as it maintains exact
# position it can represnt sting of values

x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):

	#hidden layer has two variables in dictionary weights 784 by the number of nodes

# biggest benefits of bias is if all input is zero..bias comes in and it can help fire neuron
	
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
						'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
						'biases':tf.Variable(tf.random_normal([n_classes]))}

#input_data * weights + bias
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
#below id threshold function
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
#below id threshold function
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
#below id threshold function
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
	return output


def train_neural_network(x):
	prediction = neural_network_model(x)
 	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#above we are comparing difference between actual and predicted
#now next step is to minimize that difference or cost
#default learning rate 0.01
	optimizer = tf.train.AdamOptimizer().minimize(cost)

#feed forward + backprop cycles
	hm_epochs = 10 

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

#so basically we are trying to optimize by modifying weights
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer,cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_loss += c
			print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss)

		correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		#tf.argmax return max value in that array

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))








train_neural_network(x)












