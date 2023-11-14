import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.image import ImageDataGenerator


def load_data():
    # Load the CSV data
    train_data = pd.read_csv('data/sudoku_dataset/train.csv')
    test_data = pd.read_csv('data/sudoku_dataset/test.csv')

    return train_data, test_data


def preprocess_data(train_data, test_data):
    # Extract features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape the labels for compatibility with TensorFlow
    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    datagen.fit(X_train.reshape(-1, 9, 9, 1))

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), datagen


if __name__ == "__main__":
    # Example usage
    train_data, test_data = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, datagen = preprocess_data(train_data, test_data)

    # Print shapes to verify
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Example of using data augmentation
    augmented_images = datagen.flow(X_train.reshape(-1, 9, 9, 1), y_train, batch_size=1)[0][0]

    # Print the shape of augmented images
    print("Augmented Images shape:", augmented_images.shape)
