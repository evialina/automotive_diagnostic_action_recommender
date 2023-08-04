# # Claims & Vehicle Fault-based Diagnostic Action Prediction (CVFDA) Model
# This component uses a vehicle's fault and claim history to predict the most suitable diagnostic actions to address a particular fault. This prediction doesn't consider the sequence in which the actions should be executed.
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

import keras
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Input, Flatten, Dense, Layer, Dropout, BatchNormalization, MaxPooling2D, Reshape
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from sklearn.metrics import ndcg_score
from model_utils import save_plot_accuracy_loss, save_model

# Constants
MODEL_NAME = 'cvf_da'
EMBEDDING_DIM = 50
BATCH_SIZE = 128
EPOCHS = 2
VALIDATION_SPLIT = 0.2
SEED = 42
CATEGORICAL_FEATURES = ['model', 'modelyear', 'driver', 'plant', 'engine', 'transmission', 'module', 'dtcbase', 'faulttype',
                        'dtcfull', 'year', 'month', 'dayOfWeek', 'weekOfYear', 'season', 'i_original_vfg_code',
                        'softwarepartnumber', 'hardwarepartnumber', 'i_p_css_code', 'i_original_ccc_code',
                        'i_original_function_code', 'i_original_vrt_code', 'i_current_vfg_code', 'i_current_function_code',
                        'i_current_vrt_code',	'i_cpsc_code', 'i_cpsc_vfg_code', 'i_css_code', 'v_transmission_code',
                        'v_drive_code', 'v_engine_code', 'ic_repair_dealer_id', 'ic_eng_part_number', 'ic_serv_part_number',
                        'ic_part_suffix', 'ic_part_base', 'ic_part_prefix', 'ic_causal_part_id', 'ic_repair_country_code']
NUMERICAL_FEATURES = ['elapsedTimeSec', 'timeSinceLastActivitySec', 'odomiles', 'vehicleAgeAtSession',
                      'daysSinceWarrantyStart', 'i_mileage', 'i_time_in_service', 'i_months_in_service']


# # Convolution with Cross Convolutional Filters
# @keras.saving.register_keras_serializable('models') # for tensorflow >=12.3
@keras.utils.register_keras_serializable('models') # for tensorflow <=12.2
class CrossConv2D(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(CrossConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1 = tf.keras.layers.Conv2D(filters, (kernel_size[0], 1), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, (1, kernel_size[1]), activation='relu', padding='same')

    def call(self, inputs):
        conv1_output = self.conv1(inputs)
        conv2_output = self.conv2(inputs)
        return conv1_output + conv2_output

    def get_config(self):
        base_config = super().get_config()
        config = {"filters": self.filters, "kernel_size": self.kernel_size}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def load_data(path):
    data_df = pd.read_csv(path)
    data_df = data_df.iloc[:, 1:]  # remove index column
    return data_df


def encode_categorical_features(data_df, label_encoder):
    # Convert each categorical feature to integer encoding
    for feature in CATEGORICAL_FEATURES:
        data_df[feature] = label_encoder.fit_transform(data_df[feature])
    return data_df


def variable_embedding(data_df):
    input_layers = []
    embedding_layers = []

    for col in CATEGORICAL_FEATURES:
        num_unique_categories = data_df[col].nunique()

        # Create input layer for each category
        input_layer = Input(shape=(1,), name=f"{col}_input")
        input_layers.append(input_layer)

        # Create embedding layer for each category
        embedding = Embedding(num_unique_categories, EMBEDDING_DIM, input_length=1, name=f"{col}_embedding")(input_layer)

        # Flatten the embedding layer
        embedding_flatten = Flatten()(embedding)
        embedding_layers.append(embedding_flatten)

    # Define the input layer for the numerical features
    num_input = Input(shape=(len(NUMERICAL_FEATURES),), name='numerical_input')

    # Pass the numerical inputs through a MLP
    hidden1 = Dense(128, activation='relu')(num_input)
    hidden2 = Dense(64, activation='relu')(hidden1)
    num_output = Dense(32, activation='relu')(hidden2)  # This is the embedding of the continuous features

    # Append num_output to the list of embedding layers
    embedding_layers.append(num_output)

    return input_layers, num_input, embedding_layers


def reshaping_inputs(embeddings_concat):
    print(f'Original embeddings shape: {embeddings_concat.shape}')
    total_features = embeddings_concat.shape[1] # the total number of features after concatenation
    sqrt_features = int(np.ceil(np.sqrt(total_features))) # the nearest square number greater than total_features

    flattened = tf.keras.layers.Flatten()(embeddings_concat)
    dense_for_reshape = tf.keras.layers.Dense(sqrt_features*sqrt_features, activation='relu')(flattened)

    embeddings_reshaped = tf.keras.layers.Reshape((sqrt_features, sqrt_features, 1))(dense_for_reshape)
    print(f'Reshaped embeddings shape: {embeddings_reshaped.shape}')
    return embeddings_reshaped


def compile_model(input_layers, num_input, output_layer):
    model = Model(inputs=input_layers + [num_input], outputs=[output_layer])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',
                                                                              tf.keras.metrics.AUC(),
                                                                              tf.keras.metrics.Precision(),
                                                                              tf.keras.metrics.Recall()])
    plot_model(model, to_file='fixtures/models/cvf-da_layers.png', show_shapes=True, show_layer_names=True)
    return model


def train_model(model, train_input, y_train, test_input, y_test):
    early_stopping_monitor = EarlyStopping(
        monitor='accuracy',
        min_delta=0.001,  # minimum change to qualify as an improvement
        patience=5,  # number of epochs with no improvement after which training will be stopped
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    history = model.fit(
        train_input,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(test_input, y_test),
        callbacks=[early_stopping_monitor]
    )

    return history


def evaluate_model(model, test_input, y_test):
    # Evaluate the performance of the model on the test data
    loss, accuracy, auc, precision, recall = model.evaluate(test_input, y_test)
    f1 = 2 * (recall * precision / (recall + precision))
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    print(f"Test AUC-ROC: {auc}")
    print(f"Test Precision: {precision}")
    print(f"Test Recall: {recall}")
    print(f"Test F1: {f1}")

    # Calculate the predicted class as the one with highest probability
    y_pred = model.predict(test_input)
    # y_pred_class = np.argmax(y_pred, axis=1)
    # y_test_classes = np.argmax(y_test, axis=1)

    # Calculate metrics
    ndcg = ndcg_score(y_test, y_pred)
    print(f"Test NDCG: {ndcg}")


def main():
    data_df = load_data('./data_out/prepared_data_half.csv')
    with open('fixtures/label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    data_df = encode_categorical_features(data_df, label_encoder)

    input_layers, num_input, embedding_layers = variable_embedding(data_df)
    embeddings_concat = tf.keras.layers.concatenate(embedding_layers)
    embeddings_reshaped = reshaping_inputs(embeddings_concat)

    # Model
    cross_conv1 = CrossConv2D(filters=32, kernel_size=(3, 3))(embeddings_reshaped)
    batch_norm1 = BatchNormalization()(cross_conv1)
    activation1 = tf.keras.activations.relu(batch_norm1)
    pooling1 = MaxPooling2D(pool_size=(2, 2))(activation1)
    dropout1 = Dropout(0.25)(pooling1)

    cross_conv2 = CrossConv2D(filters=64, kernel_size=(3, 3))(dropout1)
    batch_norm2 = BatchNormalization()(cross_conv2)
    activation2 = tf.keras.activations.relu(batch_norm2)
    pooling2 = MaxPooling2D(pool_size=(2, 2))(activation2)
    dropout2 = Dropout(0.25)(pooling2)

    flatten = Flatten()(dropout2)
    dense1 = Dense(256, activation='relu')(flatten)
    dropout3 = Dropout(0.5)(dense1)

    num_unique_otxsequence = data_df['otxsequence'].nunique()
    output_layer = Dense(num_unique_otxsequence, activation='softmax')(dropout3)

    model = compile_model(input_layers, num_input, output_layer)

    # Prepare training and testing data
    target = to_categorical(data_df['otxsequence'])
    features = data_df.drop(columns='otxsequence')
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    train_input = [X_train[feature].values for feature in CATEGORICAL_FEATURES] + [X_train[NUMERICAL_FEATURES].values]
    test_input = [X_test[feature].values for feature in CATEGORICAL_FEATURES] + [X_test[NUMERICAL_FEATURES].values]

    # Training the model
    history = train_model(model, train_input, y_train, test_input, y_test)
    save_model(model, MODEL_NAME)
    save_plot_accuracy_loss(history, 'CVF-DA', f'{MODEL_NAME}/{MODEL_NAME}_training')

    # Evaluate the model
    evaluate_model(model, test_input, y_test)


if __name__ == "__main__":
    main()
