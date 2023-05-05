import datag
import tensorflow as tf
import numpy as np

from sklearn import svm




def mue_loss(y_true, y_pred, rho_pos, rho_neg):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    loss_pos = (1 - rho_pos) * tf.nn.relu(1 - y_true * y_pred)
    loss_neg = (1 - rho_neg) * tf.nn.relu(1 + y_true * y_pred)
    numerator = tf.reduce_mean(loss_pos - loss_neg)
    denominator = 1 - rho_pos - rho_neg
    return numerator / denominator


def create_model1(train_data, test_data, true_dict, rho_pos, rho_neg):
  x_train = [(i[1],i[2]) for i in train_data]
  y_train = [int(i[0]) for i in train_data]

  x_test = [(i[1],i[2]) for i in test_data]
  # y_test = [int(i[0]) for i in test_data]
  y_test = [true_dict[i[1],i[2]] for i in test_data]

  # svm_classifier = svm.SVC(loss = lambda y_true, y_pred: partial_loss(y_ture, y_pred, rho_plus, rho_minus, rho_y, rho_neg_y))
  # svm_classifier = svm.SVC()
  # svm_classifier.fit(x_train, y_train)

  # flag = 0
  class MyCallback(tf.keras.callbacks.Callback):
    flag = 0
    count = 0
    def __init__(self, threshold_acc):
        super(MyCallback, self).__init__()
        self.threshold_acc = threshold_acc
        self.reset_epochs = 3
        
    def on_epoch_end(self, epoch, logs={}):
        if (self.flag == 0): val_acc = logs.get('val_accuracy')
        else: val_acc = 1
        #if epoch % self.reset_epochs == 0 and epoch > 0:
          #print(val_acc)
          # print(self.threshold_acc / 2 * 1.1)
          #print(val_acc > self.threshold_acc / 2 * 1.1)
        if (val_acc > (self.threshold_acc / 2 * 1.1)):
            val_acc = 1
            self.flag = 1
            self.count += 1
            # print("\nReached {0}% validation accuracy, stopping training.".format(self.threshold_acc*100))
            # print("\nReached {0}% validation accuracy, stopping training.".format(self.threshold_acc*100))
            if self.count % self.reset_epochs == 0 and epoch > 0:
              # flag = 1
              self.model.stop_training = True
        else:
            if epoch % self.reset_epochs == 0 and epoch > 0:
            # print(val_acc)
            # print("\nResetting model with random weights")
            #  self.model.set_weights([np.array([random.random() for _ in range(layer.shape[1])]) for layer in self.model.get_weights()])
              tmp_model = create_model3(train_data, rho_pos, rho_neg)
              weights = []
              for layer in tmp_model.get_weights():
                layer_shape = layer.shape
                layer_weights = np.random.randn(*layer_shape)
                
                weights.append(layer_weights)
              self.model.set_weights(weights)
              # self.model = create_model1(data)  # reinitialize model with random weights

  callback = [MyCallback(threshold_acc = 1 * (1 - rho_pos) * (1 - rho_neg) * 1.3)]
  learning_rate = 0.1
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  # print((np.array(x_train).shape[1],))
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=(np.array(x_train).shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='tanh'),
    tf.keras.layers.Lambda(lambda x: tf.math.sign(x))
  ])

  model.compile(optimizer='adam',
              # loss= lambda y_ture, y_pred: partial_loss(y_ture, y_pred, rho_plus, rho_minus),
              loss= lambda y_true, y_pred: mue_loss(y_true, y_pred, rho_pos, rho_neg),
              metrics=['accuracy'])

  model.fit(np.array(x_train), np.array(y_train), epochs= 15, verbose=0, validation_data=(np.array(x_test), np.array(y_test)), callbacks=[callback])    
  return model

def create_model2(train_data, test_data, true_dict, rho_pos, rho_neg):
  x_train = [(i[1],i[2]) for i in train_data]
  y_train = [i[0] for i in train_data]

  x_test = [(i[1],i[2]) for i in test_data]
  y_test = [true_dict[i[1],i[2]] for i in test_data]

  class_weights = {1: 1 - (1 - rho_pos + rho_neg) / 2, -1: (1 - rho_pos + rho_neg) / 2}
  # svm_classifier = svm.SVC(loss = lambda y_true, y_pred: partial_loss(y_ture, y_pred, rho_plus, rho_minus, rho_y, rho_neg_y))
  svm_classifier = svm.SVC(class_weight=class_weights)
  svm_classifier.fit(x_train, y_train)

  return svm_classifier

def create_model3(train_data, rho_pos, rho_neg):
  x_train = [(i[1],i[2]) for i in train_data]
  y_train = [int(i[0]) for i in train_data]

  model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=(np.array(x_train).shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='tanh'),
    tf.keras.layers.Lambda(lambda x: tf.math.sign(x))

    
  ])

  model.compile(optimizer='adam',
              # loss= lambda y_ture, y_pred: partial_loss(y_ture, y_pred, rho_plus, rho_minus),
              loss= lambda y_true, y_pred: mue_loss(y_true, y_pred, rho_pos, rho_neg),
              metrics=['accuracy'])

  model.fit(x_train, y_train, epochs= 0, verbose=0)
  return model



