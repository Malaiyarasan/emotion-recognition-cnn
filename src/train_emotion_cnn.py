import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report


DATA_DIR = Path("data/emotions")
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15


def build_model(input_shape, num_classes: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} not found. Please create data/emotions/ with one subfolder per emotion."
        )

    # Image augmentation and generators
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    num_classes = train_gen.num_classes
    input_shape = IMG_SIZE + (3,)  # RGB images

    model = build_model(input_shape, num_classes)
    model.summary()

    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
    )

    # Evaluation with classification report
    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = preds.argmax(axis=1)
    y_true = val_gen.classes
    class_indices = {v: k for k, v in val_gen.class_indices.items()}

    target_names = [class_indices[i] for i in range(num_classes)]
    print("Classification report:\n")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Optional: save model
    model.save("emotion_cnn_model.h5")
    print("Model saved to emotion_cnn_model.h5")


if __name__ == "__main__":
    main()
