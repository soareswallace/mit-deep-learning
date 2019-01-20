import matplotlib.pyplot as plt

def plot_images(features, labels):
	plt.figure(figsize=(10,2))
	for i in range(5):
		plt.subplot(1, 5, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(features[i], cmap=plt.cm.binary)
		plt.xlabel(labels[i])
		plt.show()