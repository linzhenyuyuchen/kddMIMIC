import os
import numpy as np
import SimpleITK as sitk
################################################
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
################################################
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold, train_test_split
################################################
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
################################################
import keras
from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping
################################################
from .nnmodels.densenet3d import DenseNet3D_FCN, DenseNet3DImageNet121

################################################


class customDataGenerator(keras.utils.Sequence):

    def __init__(self, ct_filenames, roi_filenames, labels, batch_size):
        self.ct_filenames = ct_filenames
        self.roi_filenames = roi_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.ct_filenames) / float(self.batch_size))).astype(np.int)

    # TODO
    def get_array(self, path):
        # return np.load(path)
        return self.get_array_from_itk(path)
        
    def get_array_from_itk(self, path):
        itk_img = sitk.ReadImage(path)
        img_array = sitk.GetArrayFromImage(itk_img)
        return img_array

    def concateMask(self, image_path, mask_path):
        img = self.get_array(image_path)
        mask = self.get_array(mask_path)
        con = np.concatenate((img, mask))
        con = con[np.newaxis, :]
        return con.transpose((1,2,3,0))

    def __getitem__(self, idx):
        batch_x1 = self.ct_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x2 = self.roi_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        lb = []
        for i in batch_y:
            tmp = [0, 0, 0]
            tmp[i] = 1
            lb.append(tmp)
        return np.array([self.concateMask(x1, x2) for x1, x2 in zip(batch_x1, batch_x2)]), np.array(lb)

class nnHelper():
    def __init__(self, n_class=3, n_fold=5, output_dir="./log/"):
        self.n_class = n_class
        self.nfold = n_fold
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def getModel(self, input_shape=(96, 160, 160, 1)):# (1, 60, 256, 256)
        """
        model = DenseNet3DImageNet121(input_shape,classes=self.n_class)       
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])
        model.summary()
        """
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(
            input_shape), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
        model.add(Activation('softmax'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
        model.add(Dropout(0.25))

        model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
        model.add(Activation('relu'))
        model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
        model.add(Activation('softmax'))
        model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_class, activation='softmax'))

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])
        model.summary()
        #"""
        # plot_model(model, to_file=os.path.join(self.output_dir, 'model.png'), show_shapes='True')
        return model

    def splitFold(self):
        print(f"splitting dataset into {self.nfold} folds..")
        kf = StratifiedShuffleSplit(n_splits=self.nfold, test_size=0.2, random_state=0)
        idx_trva_list = []
        idx_te_list = []
        for idx_tr, idx_te in kf.split(self.labels, self.labels):
            idx_trva_list.append(idx_tr)
            idx_te_list.append(idx_te)
        idx_list = np.empty([self.nfold, 3], dtype=object)
        for i in range(self.nfold):
            idx_list[i][0] = np.setdiff1d(idx_trva_list[i], idx_te_list[(i + 1) % self.nfold], True)
            idx_list[i][1] = idx_te_list[(i + 1) % self.nfold]
            idx_list[i][2] = idx_te_list[i]
        return idx_list

    def evaluation(self, pred, gt):
        print("confusion matrix:")
        print(metrics.confusion_matrix(gt, pred))
        print("accuracy on whole set:", metrics.accuracy_score(gt, pred))

    def plot_history(self, history):
        plt.plot(history.history['accuracy'], marker='.')
        plt.plot(history.history['val_accuracy'], marker='.')
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.grid()
        plt.legend(['accuracy', 'val_accuracy'], loc='lower right')
        plt.savefig(os.path.join(self.output_dir, 'model_accuracy.png'))
        plt.close()

        plt.plot(history.history['loss'], marker='.')
        plt.plot(history.history['val_loss'], marker='.')
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(['loss', 'val_loss'], loc='upper right')
        plt.savefig(os.path.join(self.output_dir, 'model_loss.png'))
        plt.close()

    def save_history(self, history):
        loss = history.history['loss']
        acc = history.history['accuracy']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_accuracy']
        nb_epoch = len(acc)

        with open(os.path.join(self.output_dir, 'result.txt'), 'w') as fp:
            fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
            for i in range(nb_epoch):
                fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                    i, loss[i], acc[i], val_loss[i], val_acc[i]))

    def cv_train(self, img_paths, roi_paths, labels, batch_size, epochs):
        self.img_paths = np.array(img_paths)
        self.roi_paths = np.array(roi_paths)
        self.labels = np.array(labels)
        print("starting cv-train..")
        idx_list = self.splitFold()
        pred_y_all = []
        global_y_all = []
        n_fold = 0
        for idx_tr, idx_va, idx_te in idx_list:
            # Build dataset
            n_fold += 1
            # idx_trva = np.concatenate([idx_tr, idx_va])
            # ct_filenames_train, roi_filenames_train, labels_train = X[idx_trva], y[idx_trva]
            ct_filenames_train, roi_filenames_train, labels_train = self.img_paths[idx_tr], self.roi_paths[idx_tr], self.labels[idx_tr]
            ct_filenames_valid, roi_filenames_valid, labels_valid = self.img_paths[idx_va], self.roi_paths[idx_va], self.labels[idx_va]
            ct_filenames_test, roi_filenames_test, labels_test = self.img_paths[idx_te], self.roi_paths[idx_te], self.labels[idx_te]

            # earlystopping_callback = EarlyStopping(monitor='accuracy', verbose=1, min_delta=0.5, patience=5, baseline=None)
            model = self.getModel()
            # history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), batch_size=batch_size,
            #                     epochs=epochs, verbose=1, shuffle=True)  # , callbacks=[earlystopping_callback]

            my_train_batch_generator = customDataGenerator(ct_filenames_train, roi_filenames_train, labels_train, batch_size)
            my_valid_batch_generator = customDataGenerator(ct_filenames_valid, roi_filenames_valid, labels_valid, batch_size)
            my_test_batch_generator = customDataGenerator(ct_filenames_test, roi_filenames_test, labels_test, 1)
            history = model.fit_generator(generator=my_train_batch_generator,
                     steps_per_epoch = int(len(ct_filenames_train) // batch_size),
                     epochs = epochs,
                     verbose = 1,
                     validation_data = my_valid_batch_generator,
                     validation_steps = int(len(ct_filenames_valid) // batch_size))

            pred_y = np.argmax(model.predict_generator(my_test_batch_generator), axis=1)
            pred_y_all.extend(pred_y.tolist())
            global_y = np.argmax(labels_test, axis=1)
            global_y_all.extend(global_y.tolist())

            # loss, acc = model.evaluate(X_te, y_te, verbose=0)
            # print('Test loss:', loss)
            # print('Test accuracy:', acc)

            self.plot_history(history)
            self.save_history(history)

            model_json = model.to_json()
            with open(os.path.join(self.output_dir, '3dcnnmodel.json'), 'w') as json_file:
                json_file.write(model_json)
            model.save_weights(os.path.join(self.output_dir, '3dcnnmodel.hd5'))

        self.evaluation(pred_y_all, global_y_all)



    def train(self, img_paths, roi_paths, labels, batch_size, epochs):
        print("starting train..")
        
        ct_filenames_train, roi_filenames_train, labels_train = img_paths, roi_paths, labels

        self.model = self.getModel()

        my_train_batch_generator = customDataGenerator(ct_filenames_train, roi_filenames_train, labels_train, batch_size)

        history = self.model.fit_generator(generator=my_train_batch_generator,
                    steps_per_epoch = int(len(ct_filenames_train) // batch_size),
                    epochs = epochs,
                    verbose = 1)

        self.plot_history(history)
        self.save_history(history)

        model_json = self.model.to_json()
        with open(os.path.join(self.output_dir, '3dcnnmodel.json'), 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(self.output_dir, '3dcnnmodel.hd5'))
        
    def test(self, img_paths, roi_paths, labels):
        print("starting test..")
        
        ct_filenames_test, roi_filenames_test, labels_test = img_paths, roi_paths, labels
        my_test_batch_generator = customDataGenerator(ct_filenames_test, roi_filenames_test, labels_test, 1)

        pred_y_all = []
        global_y_all = []

        pred_y = np.argmax(self.model.predict_generator(my_test_batch_generator), axis=1)
        pred_y_all.extend(pred_y.tolist())
        global_y = np.argmax(labels_test, axis=1)
        global_y_all.extend(global_y.tolist())

        self.evaluation(pred_y_all, global_y_all)








