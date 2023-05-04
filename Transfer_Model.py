from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import numpy as np
import seaborn as sns
import pickle
from tensorflow.keras.callbacks import Callback


class LossCallback(Callback):
    def __init__(self, threshold):
        super(LossCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        if loss is not None and loss > self.threshold:
            print(f'\nLoss {loss} is too high. Stopping training.')
            self.model.stop_training = True


if __name__ == '__main__':
    directory = "compressed_dataset"

    new_height = 224
    new_width = 224
    input_shape = (new_height, new_width, 3)
    num_classes = 124

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(directory=directory,
                                                        target_size=input_shape[:2], batch_size=32, subset='training')
    validation_generator = train_datagen.flow_from_directory(directory=directory,
                                                             target_size=input_shape[:2], batch_size=32,
                                                             subset='validation')

    # Load the pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(new_height, new_width, 3))

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add new layers on top of the pre-trained model
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Compute class weights
    train_labels = train_generator.classes
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)

    # Convert class weights to a dictionary
    class_weight_dict = dict(enumerate(class_weights))

    threshold = 100  # set your threshold value here
    loss_callback = LossCallback(threshold)

    # Fit the model with class weights
    history = model.fit(train_generator, epochs=30, validation_data=validation_generator,
                        class_weight=class_weight_dict, callbacks=[loss_callback])

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Plot the training and validation loss over epochs
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Get the true labels and predicted labels
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

    # Save the model to a file
    model.save('model.h5')

    # Save the training history to a file
    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
