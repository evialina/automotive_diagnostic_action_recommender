import pickle

import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
from cvf_da_model_mod import encode_categorical_features, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from data_preprocessing import process_data_for_predictions


def main():
    # Load the trained models
    cvf_da_model = keras.models.load_model('models_test/cvf-da_20230801_181457.keras')

    # the data that we are generating recommendations for
    prepared_data = process_data_for_predictions('manual_test_data/manual_test_diag_data.csv')
    prepared_data = encode_categorical_features(prepared_data, LabelEncoder())

    # ############# CVF-DA MODEL #############
    # Prepare input
    prepared_data_input = [prepared_data[feature].values for feature in CATEGORICAL_FEATURES] + \
                          [prepared_data[NUMERICAL_FEATURES].values]
    cvf_da_predictions = cvf_da_model.predict(prepared_data_input)

    # The output will be the probabilities for each class.
    # To convert these probabilities into actual class predictions, take the diagnostic action with
    # the highest probability.
    predicted_actions = np.argmax(cvf_da_predictions, axis=1)

    # Convert these encoded class labels back to the original labels,
    # use the LabelEncoder that was used to fit the 'otxsequence'.
    with open('fixtures/label_encoder.pkl', 'rb') as file:
        loaded_label_encoder = pickle.load(file)
    predicted_actions = loaded_label_encoder.inverse_transform(predicted_actions)
    print(prepared_data_input)
    print(f'predicted_actions {predicted_actions}')

    # ############# DAS MODEL #############
    with open('models_test/da_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # predicted_actions_uq = ', '.join(list(predicted_actions)) # Use a set to get unique actions
    predicted_actions_uq = ', '.join(list(set(predicted_actions))) # Use a set to get unique actions

    print(f'predicted_actions_uq {predicted_actions_uq}')
    tokenized_input = tokenizer.texts_to_sequences([predicted_actions_uq])
    # Convert to the correct input shape
    das_input = np.array(tokenized_input).reshape((1, -1))
    print(f'das_input {das_input}')

    das_model = keras.models.load_model('models_test/das_20230801_185859.h5')
    final_prediction = das_model.predict(das_input)
    predicted_sequence_indices = np.argmax(final_prediction, axis=-1)
    predicted_actions = tokenizer.sequences_to_texts(predicted_sequence_indices)

    print(f'final_prediction {predicted_actions}')


if __name__ == "__main__":
    main()

