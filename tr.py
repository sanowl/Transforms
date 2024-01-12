import tensorflow as tf
import numpy as np

# Define a custom layer for the Spatial Transformer Network
class SpatialTransformerNetwork(tf.keras.layers.Layer):
    def __init__(self, output_size):
        super(SpatialTransformerNetwork, self).__init__()
        self.output_size = output_size

    def build(self, input_shape):
        # Define the Localization Network
        self.localization_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (7, 7), activation='relu', input_shape=input_shape[1:]),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(6, activation='linear')
        ])
        self.localization_net.build(input_shape)
        self.localization_net.summary()

    def call(self, inputs):
        # Apply the Localization Network to predict transformation parameters
        theta = self.localization_net(inputs)
        theta = tf.keras.layers.Reshape((2, 3))(theta)
        
        # Apply the Spatial Transformer to transform the input data
        output = tf.keras.layers.SpatialTransformer()([inputs, theta])
        return output

# Create a deeper neural network architecture
def create_advanced_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        SpatialTransformerNetwork(output_size=(32, 32)),  # STN Layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Example usage
input_shape = (64, 64, 3)  # Input shape of your data
num_classes = 10  # Number of classes in your dataset

# Create an instance of the advanced model
model = create_advanced_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load and preprocess your dataset (e.g., CIFAR-10)
# Replace this with your dataset loading and preprocessing code
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize pixel values to [0, 1]

# Define a learning rate schedule for training
def lr_schedule(epoch):
    if epoch < 50:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Define early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Train the model with callbacks
model.fit(X_train, y_train, epochs=150, batch_size=64, validation_split=0.2,
          callbacks=[lr_scheduler, early_stopping])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Use the trained model for image classification
# Replace X_new_data with your new data for predictions
predictions = model.predict(X_new_data)
# Now, 'predictions' contains the predicted class probabilities for your new data
