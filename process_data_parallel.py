import os
import pickle
import multiprocessing as mp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import load_data, \
    remove_outlier_diagnostic_activities, remove_duplicates, process_data_for_training, save_csv

global lock
lock = mp.Lock()


# Split large csv file into smaller files
def split_csv(filename, new_filename, chunk_size):
    if not os.path.exists('data/chunks'):
        os.makedirs('data/chunks')

    chunk_no = 1
    for chunk in pd.read_csv(filename, chunksize=chunk_size, low_memory=False):
        chunk.to_csv(f'data/chunks/{new_filename}_chunk{chunk_no}.csv', index=False)
        chunk_no += 1


# Process and save chunks in parallel
def parallel_process_and_save_chunks(function, chunk_files):
    print(f'CPU cores count: {mp.cpu_count()}\n')
    # Create a lock object to prevent concurrent writes to the same file
    with mp.Pool() as pool:
        pool.starmap(process_and_save_chunk, [(function, chunk_file) for chunk_file in chunk_files])
    print('All chunks processed and saved.\n')


# Process a single chunk and save it to the CSV
def process_and_save_chunk(function, chunk_file):
    try:
        result = process_data_for_training(chunk_file)
        with lock:
            save_csv(result, 'data_out/prepared_chunks_data.csv', True)
            print(f'Processed and saved chunk: {chunk_file}\n')
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # Uncomment split_csv line to split the diagnostic data into 2 million data row chunks
    # split_csv('data/diagnostic_all.csv', 'diagnostic', chunksize=2000000)

    # Get list of diagnostic data chunk files
    base_path = 'data/chunks_2m'
    diagnostic_chunk_files = [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('.csv')]

    # Process data chunks
    parallel_process_and_save_chunks(process_data_for_training, diagnostic_chunk_files)

    # Process merged chunks of data to remove duplicates and outlier diagnostic activities
    prepared_data_chunks_df = load_data('data_out/prepared_chunks_data.csv')

    prepared_data_df = remove_outlier_diagnostic_activities(prepared_data_chunks_df)
    prepared_data_df = remove_duplicates(prepared_data_df).copy()

    # Encode target 'otxsequence' feature using LabelEncoder
    label_encoder = LabelEncoder()
    prepared_data_df.loc[:, 'otxsequence'] = label_encoder.fit_transform(prepared_data_df['otxsequence'])

    # Saving the encoder
    with open('fixtures/label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)

    # Save processed data
    save_csv(prepared_data_df, 'data_out/prepared_data.csv', False)


if __name__ == "__main__":
    main()
