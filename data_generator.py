# train a generative adversarial network on a one-dimensional function
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
import tensorflow as tf
from matplotlib import pyplot
import os 
import numpy as np

# define the standalone discriminator model
def define_discriminator(n_inputs=234):
	model = Sequential()
	model.add(tf.keras.Input(shape=(n_inputs,)))
	model.add(tf.keras.layers.Dense(234))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))	
	model.add(tf.keras.layers.Dense(32))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	model.add(tf.keras.layers.Dense(32))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
	#model.add(Dense(1, activation='sigmoid'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=234):
	model = Sequential()
	#model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(tf.keras.Input(shape=(latent_dim,)))
	model.add(tf.keras.layers.Dense(64))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.ReLU())	
	model.add(tf.keras.layers.Dense(32))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.ReLU())	
	model.add(tf.keras.layers.Dense(32))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.ReLU())	
	model.add(tf.keras.layers.Dense(32))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.ReLU())	
	model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
	#model.add(Dense(n_outputs, activation='linear'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# generate n real samples with class labels
# def generate_real_samples(n):
# 	# generate inputs in [-0.5, 0.5]
# 	X1 = rand(n) - 0.5
# 	# generate outputs X^2
# 	X2 = X1 * X1
# 	# stack arrays
# 	X1 = X1.reshape(n, 1)
# 	X2 = X2.reshape(n, 1)
# 	X = hstack((X1, X2))
# 	# generate class labels
# 	y = ones((n, 1))
# 	return X, y

def get_real_samples(n):
    #############################################################################################################
        #msu data analysis
    #############################################################################################################     
	path_to_U87_1 = os.path.join('data_thz/U87_1/')
	path_to_U87_2 = os.path.join('data_thz/U87_2/')
	path_to_U87_3 = os.path.join('data_thz/U87_3/')
	path_to_U87_4 = os.path.join('data_thz/U87_4/')
	path_to_control_1 = os.path.join('data_thz/Control_1/')
	path_to_control_2 = os.path.join('data_thz/Control_2/')
	path_to_control_3 = os.path.join('data_thz/Control_3/')
	path_to_control_4 = os.path.join('data_thz/Control_4/')             


	frequencies = []
	U87_1 = []
	U87_2 = []
	U87_3 = []
	U87_4 = []
	control_1 = []
	control_2 = []
	control_3 = []
	control_4 = []

	frequency_path_thz = 'data_thz/freq.txt'
	frequencies = np.genfromtxt(frequency_path_thz, delimiter=',')

#############################################################################################################    
	# Чтение данных из файлов
#############################################################################################################    
	cutoff_freq = 234 
	# U87_1
	file_list = os.listdir(path_to_U87_1)
	X=[]
	for fn in file_list:
		with open(os.path.join(path_to_U87_1, fn)) as ref:
			result = np.genfromtxt(ref, delimiter=",")
			X.append(result[:cutoff_freq,1])
	y = ones((n, 1))
	return np.asarray(X), y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n, 1))
	return X, y

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=5):
	# prepare real samples
	#x_real, y_real = generate_real_samples(n)
	x_real, y_real = get_real_samples(n)
	# evaluate discriminator on real examples
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
	# evaluate discriminator on fake examples
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print(epoch, acc_real, acc_fake)
	# scatter plot real and fake data points
	pyplot.plot(x_real[0, :], color='red')
	pyplot.plot(x_fake[0, :], color='blue')
	#pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
	#pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
	pyplot.show()

# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=5, n_eval=10000):
	# determine half the size of one batch, for updating the discriminator
	#half_batch = int(n_batch / 2)
	train_batch = 5
	# manually enumerate epochs
	for i in range(n_epochs):
		# prepare real samples
		#x_real, y_real = generate_real_samples(half_batch)
		x_real, y_real = get_real_samples(n_batch)
		# prepare fake examples
		x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_batch)
		# update discriminator
		d_model.train_on_batch(x_real, y_real)
		d_model.train_on_batch(x_fake, y_fake)
		# prepare points in latent space as input for the generator
		x_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_model.train_on_batch(x_gan, y_gan)
		# evaluate the model every n_eval epochs
		if (i+1) % n_eval == 0:
			summarize_performance(i, g_model, d_model, latent_dim)

# size of the latent space
#latent_dim = 5
latent_dim = 1000
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)
generator.save('generator')
discriminator.save('discriminator')
gan_model.save('gan')
print('Done')