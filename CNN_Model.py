from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, regularizers
import numpy as np
import seaborn as sns
import pickle


if __name__ == '__main__':
    directory = "compressed_dataset"

    new_height = 512
    new_width = 220
    input_shape = (new_height, new_width, 3)
    num_classes = 124

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(directory=directory,
                                                        target_size=input_shape[:2], batch_size=32, subset='training')
    validation_generator = train_datagen.flow_from_directory(directory=directory,
                                                             target_size=input_shape[:2], batch_size=32,
                                                             subset='validation')

    model = models.Sequential()

    # add convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # add flatten and dense layers with kernel regularization
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))

    model.summary()  # print the model summary

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Compute class weights
    train_labels = train_generator.classes
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)

    # Convert class weights to a dictionary
    class_weight_dict = dict(enumerate(class_weights))

    # Fit the model with class weights
    history = model.fit(train_generator, epochs=30, validation_data=validation_generator) #,class_weight=class_weight_dict)

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
    model.save('model_CNN_AR.h5')

    # Save the training history to a file
    with open('history_CNN_AR.pkl', 'wb') as f:
        pickle.dump(history.history, f)
