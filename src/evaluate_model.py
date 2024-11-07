from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
test_dir = 'C:/3RD-YEAR/INDIVIDUAL PROJECT/Skin Classification/data/test'  # Update with actual path
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
MODEL_PATH = 'C:/3RD-YEAR/INDIVIDUAL PROJECT/Skin Classification/models/CNNFirst.h5'  # Update with your model's path

# Load the trained model
model = load_model(MODEL_PATH)

# Data generator for test data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.2f}")
