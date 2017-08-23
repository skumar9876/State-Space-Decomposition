import numpy as np
import pickle
import matplotlib.pyplot as plt

def make_plot(file_name, episode_lengths):
	locations = np.arange(len(episode_lengths))

	means = []
	error_arr = []
	episodes = []

	step_size = 200

	for i in xrange(0, len(episode_lengths), step_size):
		episodes.append(i)

		array_subsection = episode_lengths[i:i+step_size-1]

		means.append(np.sum(array_subsection) / step_size)
		error_arr.append(np.std(array_subsection))

	fig1 = plt.figure()
	#plt.scatter(episodes, means)
	plt.errorbar(episodes, means, error_arr, fmt='ok', lw=1)
	fig1.savefig(file_name + '.png', dpi=fig1.dpi)
	plt.close()

                    