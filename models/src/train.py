import tensorflow as tf
from models.src.model import create_model
from models.src.preprocess import load_data, preprocess_data

def train_model():
    # Load and preprocess data
    train_data, test_data = load_data()
    (X_train, y_train), (X_val, y_val), _, datagen = preprocess_data(train_data, test_data)

    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Data augmentation flow
    train_flow = datagen.flow(X_train.reshape(-1, 9, 9, 1), y_train, batch_size=32)

    # Train the model
    model.fit(train_flow, epochs=10, validation_data=(X_val, y_val))

    # Save the trained model
    model.save('models/sudoku_solver_model.h5')

if __name__ == "__main__":
    train_model()
