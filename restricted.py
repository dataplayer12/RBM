'''
Implements a restricted Boltzmann Machine (compatible with python 2 & 3)
Notation is as per Hugo Larochelle's lecture on YouTube:
https://www.youtube.com/watch?v=MD8qXWucJBY
'''
import numpy as np
import pickle #used to save trained RBM
import os #used to load saved RBM

class RBM:
	def __init__(self,inputs_vec,alpha,H,batchsize,sigma=0.05):
		
		self.inputs=inputs_vec #input data: expected shape (n_datapoints,n_features)
		self.visible_units=self.inputs.shape[1] #n_features
		self.hidden_units=H
		self.alpha=alpha #learning rate
		self.allowedk=np.arange(10)*2 + 1 # allowed chain length k in CD-k #1 to 19
		self.sigma=sigma #stddev of initialized weights
		self.batchsize=batchsize
		self.weights=self.sigma*np.random.randn(self.visible_units,H)
		self.b=self.sigma*np.random.randn(1,H) #biases for hidden layer
		self.c=self.sigma*np.random.randn(1,self.visible_units) #biases for visible layer

	def sigmoid(self,x):
		return 1.0/(1.0+np.exp(-x))

	def get_minibatch(self):
		num_batches=self.inputs.shape[0]//self.batchsize
		for idx in range(num_batches):
			batchx=self.inputs[idx*self.batchsize:(idx+1)*self.batchsize,:]
			yield batchx

	def sampleh(self,x,get_prob=False):
		#p(hj=1|x)= sigmoid(bj+Wj.*x) #refer to Hugo Larochelle's lecture
		probabilities=self.sigmoid(x.dot(self.weights)+self.b)
		if get_prob:
			return probabilities
		vec=np.random.random(size=[x.shape[0],self.hidden_units])
		sampled = (vec<probabilities)
		return sampled

	def samplex(self,h,get_prob=False):
		#p(xk=1|h)= sigmoid(ck+h.T*W.k) #refer to Hugo Larochelle's lecture
		probabilities=self.sigmoid(self.c+self.weights.dot(h.T).T)
		if get_prob:
			return probabilities
		vec=np.random.random(size=[h.shape[0],self.visible_units])
		sampled = (vec < probabilities)
		return sampled

	def get_xtilda(self,xtilda,k):
		if k is None:
			k=np.random.choice(self.allowedk)
		
		this_x=xtilda
		for iter in range(k):
			this_h=self.sampleh(this_x,get_prob=True)
			this_x=self.samplex(this_h,get_prob=True)
			#while training we use probabilities instead of sampling binary vectors themselves
			#This makes all the difference in the world 
			#refer to Section 3 of Hinton's 'A practical guide to training RBMs'
		return this_x

	def NLL(self):
		xs=self.inputs[:100,:]
		hidden=self.sampleh(xs)
		probabilities=self.samplex(hidden,get_prob=True)
		return -np.mean(np.log(probabilities))

	def fit(self,epochs=10,k=None):
		if k is None:
			print('Training with random number  of steps')
		
		xtilda=self.inputs[:self.batchsize,:] #initialize Markov chain to x
		for e in range(epochs):
			for batchx in self.get_minibatch():
				xtilda=self.get_xtilda(xtilda,k=k) #performs persitent contrastive divergence
		
				self.weights += (self.alpha/self.batchsize)*(np.dot(self.sampleh(batchx,1).T,batchx)-np.dot(self.sampleh(xtilda,1).T,xtilda)).T
				self.b += self.alpha*(self.sampleh(batchx,1)-self.sampleh(xtilda,1)).mean(axis=0)
				self.c += self.alpha*(batchx-xtilda).mean(axis=0)

			print('Epoch {}/{}: NLL={:.5f}'.format(e+1,epochs,self.NLL()))
		print('RBM trained')

	def save(self,filename): #helper function to save trained model
		with open(filename+'.p','wb') as f:
			pickle.dump((self.weights,self.b,self.c),f)
		print('Trained RBM saved')
	
	def load(self,filename): #helper function to load saved model
		if not filename in os.listdir('./'):
			print('{} was not found in this folder'.format(filename))
		with open(filename,'rb') as f:
			try:
				self.weights,self.b,self.c = pickle.load(f,encoding='latin1')
			except:
				self.weights,self.b,self.c = pickle.load(f) #for python 2.x
		print('Trained RBM loaded')