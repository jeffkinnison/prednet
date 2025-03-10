'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from prednet_sequence_generators import CloudVolumeSegmentationSequenceGenerator
from kitti_settings import *


n_plot = 40
batch_size = 10
nt = 10

weights_file = os.path.join('cremi', 'prednet_bfly_weights.hdf5')
json_file = os.path.join('cremi', 'prednet_bfly_model.json')
test_file = os.path.join('file:///scratch0/EM_Challenge_Sets/sample_A/image')
seg_file = os.path.join('file:///scratch0/EM_Challenge_Sets/sample_A/segmentation')
RESULTS_SAVE_DIR = 'sample_A_results'
# test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = CloudVolumeSegmentationSequenceGenerator(test_file, seg_file, nt, 1, (160, 128), xy_stride=(152, 120), batch_size=batch_size, sequence_start_mode='unique', data_format=data_format, shuffle=False)
print('Create X_test')
X_test = test_generator.create_all()
print('Create X_hat')
X_hat = test_model.predict(X_test, batch_size)
# X_hat = test_model.predict_generator(test_generator, int(np.ceil(test_generator.n_blocks / batch_size)))
# if data_format == 'channels_first':
#     X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
#     X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:])**2)  # look at all timesteps except the first
mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:])**2)
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()

print(X_test.shape, X_hat.shape)

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 4*aspect_ratio))
gs = gridspec.GridSpec(4, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(np.squeeze(X_test[i, t, :, :, :-1]), interpolation='none', cmap='Greys_r')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t == 0: plt.ylabel('Actual Frame', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(np.squeeze(X_hat[i, t, :, :, :-1]), interpolation='none', cmap='Greys_r')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t == 0: plt.ylabel('Predicted Frame', fontsize=10)

        plt.subplot(gs[t + 2 * nt])
        plt.imshow(np.squeeze((X_test[i, t, :, :, -1] * test_generator.max_label)).astype(np.uint32), interpolation='none', cmap='Dark2')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t == 0: plt.ylabel('Actual Segmentation', fontsize=10)

        plt.subplot(gs[t + 3 * nt])
        plt.imshow(np.squeeze((X_hat[i, t, :, :, -1] * test_generator.max_label).astype(np.uint32)), interpolation='none', cmap='Dark2')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t == 0: plt.ylabel('Predicted Segmentation', fontsize=10)
    plt.gcf().set_size_inches(40, 16) 
    plt.savefig(plot_save_dir + 'plot_' + str(i) + '.png')
    plt.clf()
