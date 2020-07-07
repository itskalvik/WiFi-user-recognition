from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from skimage.transform import resize
from tensorflow.keras import layers
import numpy as np
import argparse
import pickle
import h5py

def get_args():
    parser = argparse.ArgumentParser(description="train resent on CSI data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="h5py data file")
    parser.add_argument("--save_path", type=str, required=True, help="h5py data file for weight")
    parser.add_argument("--history_path", type=str, required=True, help="pkl data file for history")
    parser.add_argument("--ngpu", type=int, required=True, help="number of gpus")
    args = parser.parse_args()

    return args

def identity_block(input_tensor, filters, activation_func):
    filters1, filters2 = filters
    
    x = layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters1, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, (1, 1), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)    
    return x

def conv_block(input_tensor, filters, activation_func, strides=(2, 2)):
    filters1, filters2 = filters

    x = layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters1, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, (1, 1), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=-1)(x)

    shortcut = layers.Conv2D(filters2, (1, 1), strides=strides, kernel_initializer='he_normal')(input_tensor)
    shortcut = layers.BatchNormalization(axis=-1)(shortcut)


    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def main():
    args=get_args()
    data_dir=args.data_dir
    save_path=args.save_path
    history_path=args.history_path
    ngpu=args.ngpu

    lr=1e-3
    epochs=1000
    decay=1e-3
    height = 256
    width = 2048

    hf = h5py.File(data_dir, 'r')
    train_classes = np.array(hf.get('labels')).astype(str)
    num_classes = len(train_classes)
    X_train = np.expand_dims(hf.get('X_train'), axis=-1)
    X_test = np.expand_dims(hf.get('X_test'), axis=-1)
    y_train = np.eye(num_classes)[hf.get('y_train')]
    y_test = np.eye(num_classes)[hf.get('y_test')]
    hf.close()

    X_train = np.array([resize(X_train[i], (height, width), mode='reflect', anti_aliasing=True) for i in range(X_train.shape[0])])
    X_test = np.array([resize(X_test[i], (height, width), mode='reflect', anti_aliasing=True) for i in range(X_test.shape[0])])

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, num_classes)
    
    input_layer = layers.Input(shape=(X_train.shape[1:]))

	x = layers.Conv2D(16, (7, 7),
	                  activation=None,
	                  padding='same',
	                  kernel_initializer='he_normal')(input_layer) 
	x = layers.BatchNormalization(axis=-1)(x)
	x = layers.Activation('relu')(x)
	x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

	x = conv_block(x, [16, 64], "relu")
	x = identity_block(x, [16, 64], "relu")

	x = conv_block(x, [32, 128], "relu")
	x = identity_block(x, [32, 128], "relu")

	x = conv_block(x, [64, 128], "relu")
	x = identity_block(x, [64, 128], "relu")

	x = layers.GlobalAveragePooling2D()(x)
	x = layers.Dense(num_classes, activation='softmax')(x)

    model_base = Model(inputs=input_layer, outputs=x)
    model_base.summary()
    model = multi_gpu_model(model_base, gpus=ngpu)

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr, decay=decay), metrics=['acc'])
    history = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2, batch_size=16)

    model_base.save(save_path)
    
    outfile = open(history_path,'wb')
    pickle.dump(history.history, outfile)
    outfile.close()

if __name__ == '__main__':
    main()

