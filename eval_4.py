import keras
import tensorflow as tf
import sys
import h5py
import numpy as np

clean_data_filename = str(sys.argv[1])
poisoned_data_filename = str(sys.argv[2])
model_filename = "models/bd_net.h5"
mask_filename = "models/mask_4.npy"

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def main():
    cl_x_test, cl_y_test = data_loader(clean_data_filename)
    bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

    bd_model = keras.models.load_model(model_filename)
    np_mask = np.load(mask_filename)
    feature_net = keras.Model(inputs=bd_model.input, outputs=bd_model.layers[6].output)
    interpretation_net = keras.Model(inputs=bd_model.layers[7].input, outputs=bd_model.output)
    mask = tf.Variable(np_mask, trainable=False, dtype=tf.float32)
    masked = keras.layers.Lambda(lambda x: x * mask)(feature_net.output)
    pruned_net = keras.Model(inputs=feature_net.output, outputs=interpretation_net(masked))

    activations = feature_net.predict(cl_x_test)
    y_1 = np.argmax(interpretation_net.predict(activations), axis=1)
    y_2 = np.argmax(pruned_net.predict(activations), axis=1)
    cl_label_p = np.where(y_1 == y_2, y_1, 1284)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
    print('Clean Classification accuracy:', clean_accuracy)
    
    activations = feature_net.predict(bd_x_test)
    y_1 = np.argmax(interpretation_net.predict(activations), axis=1)
    y_2 = np.argmax(pruned_net.predict(activations), axis=1)
    bd_label_p = np.where(y_1 == y_2, y_1, 1284)
    asr = np.mean(np.equal(bd_label_p, bd_y_test))*100
    print('Attack Success Rate:', asr)

if __name__ == '__main__':
    main()
