from keras.layers import Input, Dense
from keras.applications import Xception
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from MyImageDataGenerator import MyImageDataGenerator
import matplotlib.pyplot as plt
import time

# declaring some variables
imageWidth = 299
imageHeight = 299
imageDepth = 3
numberOfClasses = 257
numberOfImagesPerBatch = 32
numberOfEpochs = 100
numberOfImagesToTrain = 26752
numberOfTrainingBatches = numberOfImagesToTrain / numberOfImagesPerBatch
numberOfImagesToValidate = 3855
numberOfValidationBatches = numberOfImagesToValidate / numberOfImagesPerBatch

trainingDirectory = r'train_data'
validationDirectory = r'validation_data'

# adjust the input format to match the pre-trained model
inputFormat = Input(shape=(imageWidth, imageHeight, imageDepth))

# ------------------------- Building the Model -------------------------
# include a pre-trained model
preTrainedModel = Xception(include_top=False, weights='imagenet', input_tensor=inputFormat, pooling='avg')
# preTrainedModel = Xception(include_top=False, weights='imagenet', input_tensor=inputFormat, input_shape=inputFormat, pooling='avg', classes=256)
x = preTrainedModel.output

# freeze some layers to make them un-trainable, because reduce time and no need for that
for layer in preTrainedModel.layers:
    layer.trainable = False

# add layers to the the model (optional)

# add output layer to the model
outputLayer = Dense(numberOfClasses, activation='softmax')(x)

# decide on the optimizer
myOptimizer = SGD(lr=0.0001, momentum=0.8)  # gradient decent with low learning rate
# myOptimizer = Adam(lr=0.0001)      # adam with low learning rate

# create the model
myModel = Model(inputs=[preTrainedModel.input], outputs=[outputLayer])

# compile the model
myModel.compile(optimizer=myOptimizer, loss='categorical_crossentropy', metrics=['accuracy'])   # categorical_crossentropy iss for negative log likelihood

myModel.summary()      # print out summary of my model

# ------------------------- Creating callbacks -------------------------
# set a checkpoint to store your weights
checkpointCallback = ModelCheckpoint(filepath="./weights.hdf5", monitor="val_acc", verbose=1, save_best_only=True)

# Stop training when loss has stopped improving.
earlyStoppingCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)

# tensorBoardCallback = TensorBoard(log_dir="./logs/training_" + time.strftime("%c"), histogram_freq=0, write_graph=True,                                                    write_images=False)
reduceOnPlateauCallback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',                                                    epsilon=0.0001, cooldown=0, min_lr=0.000001)
terminateOnNaNCallback = TerminateOnNaN()

callbacks = [checkpointCallback, earlyStoppingCallback, reduceOnPlateauCallback, terminateOnNaNCallback]

# ------------------------- Creating callbacks -------------------------
# Setup Data Generator for Training and Validation
trainingDataGenerator = MyImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # don't set each sample mean to 0
    featurewise_std_normalization=True,  # divide all inputs by std of the dataset
    samplewise_std_normalization=False,  # don't divide each input by its std
    zca_whitening=False,  # don't apply ZCA whitening.
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180).
    horizontal_flip=True,  # randomly flip horizontal images.
    vertical_flip=False,  # don't randomly flip vertical images.
    zoom_range=0.1,  # slightly zoom in.
    width_shift_range=0.1,
    height_shift_range=0.1
)

validationDataGenerator = MyImageDataGenerator(
    featurewise_center=True,  # test images should have input mean set to 0 over the images.
    featurewise_std_normalization=True,  # test images should have all divided by std of the images.
    zca_whitening=False
)

# Load Data for Training and Validation
trainingDataGenerator = trainingDataGenerator.flow_from_directory(trainingDirectory,
                                                                  target_size=(imageWidth, imageHeight),
                                                                  batch_size=numberOfImagesPerBatch,
                                                                  shuffle=True)

# Loading Validation Data...
validationDataGenerator = validationDataGenerator.flow_from_directory(validationDirectory,
                                                                      target_size=(imageWidth,imageHeight),
                                                                      batch_size=numberOfImagesPerBatch,
                                                                      shuffle=False)
print("Data Set loaded!" + '\n')
print("Now Training...")

# Train Your Model



history = myModel.fit_generator(trainingDataGenerator,
                                steps_per_epoch=numberOfTrainingBatches, epochs=100, validation_data=validationDataGenerator,
                                validation_steps=numberOfValidationBatches, verbose=1, callbacks=callbacks)

myModel.save('model')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
