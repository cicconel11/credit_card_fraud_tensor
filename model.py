import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Define the dataset handler
class CreditCardFraudDataset:
    def __init__(self, filepath, test_size=0.2, random_seed=42):
        self.filepath = filepath
        self.test_size = test_size
        self.random_seed = random_seed
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.filepath)
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_seed, stratify=y)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

class TFDataFeed:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=len(X))
        self.dataset = self.dataset.batch(batch_size)

    def get_dataset(self):
        return self.dataset


class FraudDetectionModel1(tf.keras.Model):
    def __init__(self, input_shape):
        super(FraudDetectionModel1, self).__init__()
        self.dense1 = Dense(32, activation='relu', kernel_initializer='he_normal', input_shape=input_shape)
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(24, activation='relu', kernel_initializer='he_normal')
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(16, activation='relu', kernel_initializer='he_normal')
        self.dropout3 = Dropout(0.2)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        x = self.dropout3(x, training=training)
        return self.output_layer(x)

    def compile_and_train(self, X_train, y_train, X_test, y_test, epochs=12):
        # Apply resampling
        resampling_strategy = Pipeline([
            ('SMOTE', SMOTE(sampling_strategy=0.1)),
            ('RandomUnderSampler', RandomUnderSampler(sampling_strategy=0.5))
        ])
        X_resampled, y_resampled = resampling_strategy.fit_resample(X_train, y_train)

        # Convert to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled)).shuffle(10000).batch(32)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

        # Compile model
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train model
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.001)]
        history = self.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=callbacks)
        return history

class FraudDetectionModel2(tf.keras.Model):
    def __init__(self, input_shape):
        super(FraudDetectionModel2, self).__init__()  # Correct superclass reference
        self.dense1 = Dense(8, activation='relu', input_shape=input_shape)
        self.dense2 = Dense(8, activation='relu')
        self.dense3 = Dense(8, activation='relu')
        self.dense4 = Dense(8, activation='relu')
        self.dense5 = Dense(8, activation='relu')
        self.dense6 = Dense(8, activation='relu')
        self.dense7 = Dense(8, activation='relu')
        self.dense8 = Dense(8, activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)  # Ensure all layers are connected in sequence
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dense8(x)
        return self.output_layer(x)

    @staticmethod
    def compile_and_train(train_dataset, val_dataset, epochs=10):
        # Correctly instantiate FraudDetectionModel2
        model = FraudDetectionModel2(input_shape=(train_dataset.element_spec[0].shape[1],))
        optimizer = Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        return model, history


def calculate_auprc(y_true, y_scores):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return auc(recall, precision)
    except ValueError as e:
        print(f"Error calculating AUPRC: {e}")
        return np.nan  # or another value indicating the calculation was not possible

def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)

def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns  # Ensure seaborn is imported for the confusion matrix plot

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()
def clean_predictions(y_scores):
    # Check for NaN or Inf values and replace them with 0 or another suitable value
    y_scores = np.where(np.isnan(y_scores), 0, y_scores)
    y_scores = np.where(np.isinf(y_scores), 0, y_scores)
    return y_scores

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_scores = model.predict(X_test).flatten()
    y_pred_adjusted = (y_scores > threshold).astype(int)

    f1 = calculate_f1_score(y_test, y_pred_adjusted)
    auprc = calculate_auprc(y_test, y_scores)

    print(f"F1 Score: {f1}")
    print(f"AUPRC: {auprc}")

    plot_confusion_matrix(y_test, y_pred_adjusted)
def evaluate_mode(model, X_test, y_test, threshold=0.5):
    # Predict the scores using the model
    y_scores = model.predict(X_test).flatten()
    y_scores_clean = clean_predictions(y_scores)

    # Apply threshold to convert scores to binary predictions
    y_pred_adjusted = np.where(y_scores_clean > threshold, 1, 0)

    # Calculate F1 score directly using scikit-learn's utility function
    f1 = f1_score(y_test, y_pred_adjusted)

    # Calculate AUPRC
    precision, recall, _ = precision_recall_curve(y_test, y_scores_clean)
    auprc = auc(recall, precision)

    print(f"F1 Score: {f1}")
    print(f"AUPRC: {auprc}")

    # Assuming plot_confusion_matrix is correctly implemented elsewhere
    plot_confusion_matrix(y_test, y_pred_adjusted)

def find_best_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    fscore = (2 * precision * recall) / (precision + recall)
    fscore = np.nan_to_num(fscore)  # Convert NaNs to zero
    ix = np.argmax(fscore)
    return thresholds[ix], fscore[ix]

optimal_threshold = 0.891

def evaluate_model_with_threshold(model, X_test, y_test, threshold=optimal_threshold):
    y_scores = model.predict(X_test).flatten()
    y_pred_adjusted = (y_scores > threshold).astype(int)
    # Calculate and print your metrics here
    plot_confusion_matrix(y_test, y_pred_adjusted)
    print("F1 Score:", f1_score(y_test, y_pred_adjusted))

def main():
    dataset_filepath = 'creditcard1.csv'
    fraud_dataset = CreditCardFraudDataset(filepath=dataset_filepath)

    model = FraudDetectionModel1(input_shape=(fraud_dataset.X_train.shape[1],))
    history = model.compile_and_train(fraud_dataset.X_train, fraud_dataset.y_train, fraud_dataset.X_test, fraud_dataset.y_test, epochs=12)

    # Choose an appropriate threshold based on your model's performance
    threshold = 0.329
    evaluate_model_with_threshold(model, fraud_dataset.X_test, fraud_dataset.y_test, threshold)




    plot_training_history(history)

def adjust_threshold(y_scores, threshold=0.329):
    return np.where(y_scores > threshold, 1, 0)

if __name__ == "__main__":
    main()



def main():
    dataset_filepath = 'creditcard1.csv'
    dataset = CreditCardFraudDataset(filepath=dataset_filepath)

    train_dataset = TFDataFeed(dataset.X_train, dataset.y_train).get_dataset()
    test_dataset = TFDataFeed(dataset.X_test, dataset.y_test, shuffle=False).get_dataset()

    # Train the model and get the training history
    model, history = FraudDetectionModel2.compile_and_train(train_dataset, test_dataset, epochs=10)

    # Plot the training history
    print(history.history)

    plot_training_history(history)
    threshold = 0.5
    # Evaluate the model on the test dataset
    evaluate_mode(model, dataset.X_test, dataset.y_test, threshold)

if __name__ == "__main__":
    main()

