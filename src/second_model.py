import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set parameters
IMAGE_SIZE = (128, 128)  # Resize images to this size
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 30  # Increased epochs for training

# Paths to directories
train_dir = r'C:\3RD-YEAR\INDIVIDUAL PROJECT\Skin Classification\data\train'
valid_dir = r'C:\3RD-YEAR\INDIVIDUAL PROJECT\Skin Classification\data\validation'
test_dir = r'C:\3RD-YEAR\INDIVIDUAL PROJECT\Skin Classification\data\test'

# Data generators for loading and augmenting data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Define CNN model with improvements
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # Output layer with number of classes
])

# Compile the model using SGD with momentum
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(r'C:\3RD-YEAR\INDIVIDUAL PROJECT\Skin Classification\models\best_model.keras',
                                   save_best_only=True,
                                   monitor='val_accuracy')



# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model
model.save(r'C:\3RD-YEAR\INDIVIDUAL PROJECT\Skin Classification\models\CNNSecond.h5')
