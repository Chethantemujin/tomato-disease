import json
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "tomato"
MODEL_DIR = PROJECT_DIR / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
HEAD_EPOCHS = 8
FINE_TUNE_EPOCHS = 12
FINE_TUNE_LAYERS = 30

CLASS_NAMES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

DISPLAY_NAMES = {
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___healthy": "Healthy",
}


def load_dataset(split_name, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / split_name,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle,
    )


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMAGE_SIZE + (3,),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.base_model = base_model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune_model(model):
    base_model = model.base_model
    base_model.trainable = True

    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def save_validation_report(model, val_ds):
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(predictions, axis=1).tolist())

    confusion = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=len(CLASS_NAMES),
    ).numpy()

    report = {
        "classes": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
        "confusion_matrix": confusion.tolist(),
    }
    with open(MODEL_DIR / "validation_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    train_ds = load_dataset("train")
    val_ds = load_dataset("val", shuffle=False)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    print("Training classes:")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index}: {DISPLAY_NAMES[class_name]}")

    model = build_model(num_classes=len(CLASS_NAMES))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "tomato_disease_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
        ),
    ]

    print("Phase 1: training the classifier head")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=callbacks,
    )

    print("Phase 2: fine-tuning tomato leaf features")
    fine_tune_model(model)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_DIR / "tomato_disease_model.keras")
    save_validation_report(model, val_ds)

    class_info = {
        "class_names": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
        "raw_class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "confidence_threshold": 0.80,
    }
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_info, file, indent=2)

    print("Saved model to:", MODEL_DIR / "tomato_disease_model.keras")
    print("Saved class info to:", MODEL_DIR / "class_names.json")
    print("Saved validation report to:", MODEL_DIR / "validation_report.json")


if __name__ == "__main__":
    main()
