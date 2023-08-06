import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filename):
    data_df = pd.read_csv(f'{filename}',
                          dtype={'dtcbase': 'object', 'dtcfull': 'object', 'odomiles': 'float64'},
                          low_memory=False)
    return data_df


def initiate_diagnostic_consultation(diagnostic_df):
    diagnostic_df['timestamp'] = pd.to_datetime(diagnostic_df['sessiontimestamp'])
    diagnostic_df['date'] = diagnostic_df['timestamp'].dt.date
    
    diagnostic_df.sort_values(['anonymised_vin', 'date'], kind='mergesort', inplace=True)
    diagnostic_df['consultationid'] = (diagnostic_df['anonymised_vin'] != diagnostic_df['anonymised_vin'].shift()).astype(int)
    diagnostic_df['consultationid'] += (diagnostic_df['date'] != diagnostic_df['date'].shift()).astype(int)
    diagnostic_df['consultationid'] = diagnostic_df['consultationid'].cumsum()
    return diagnostic_df


def derive_vehicle_state_data(diagnostic_df):
    cols_to_keep = [col for col in diagnostic_df.columns if col not in ['otxsequence', 'date', 'sessionid', 'timestamp']]
    vehicle_current_state_df = diagnostic_df[diagnostic_df['otxsequence'] == 'G2725772'][cols_to_keep].copy()

    cols_to_keep = ['anonymised_vin', 'consultationid', 'timestamp', 'otxsequence']
    diagnostic_actions_df = diagnostic_df[diagnostic_df['otxsequence'] != 'G2725772'][cols_to_keep].copy()

    diagnostic_df = vehicle_current_state_df.merge(diagnostic_actions_df, how='inner', on=['anonymised_vin', 'consultationid'])
    return diagnostic_df


def merge_diag_warr_data(diagnostic_df, warranty_df):
    diagnostic_df['timestamp'] = pd.to_datetime(diagnostic_df['timestamp'], utc=True)
    warranty_df['i_incident_date'] = pd.to_datetime(warranty_df['i_incident_date'], utc=True)

    warranty_df = warranty_df.rename(columns={'anonymised_vin': 'warranty_anonymised_vin'})

    df_list = []
    for idx, row in diagnostic_df.iterrows():
        vin = row['anonymised_vin']
        diag_time = row['timestamp']

        mask = ((warranty_df['warranty_anonymised_vin'] == vin) &
                (warranty_df['i_incident_date'] >= diag_time) &
                (warranty_df['i_incident_date'] <= diag_time + pd.Timedelta(days=7)))

        temp_warranty_df = warranty_df[mask]

        if temp_warranty_df.empty:  # If no matching warranty claim is found, create a row with placeholders
            merged_row = row.copy()
            merged_row = pd.concat([merged_row, pd.Series([np.nan]*len(warranty_df.columns), index=warranty_df.columns)], axis=0)
        else:
            for _, warranty_row in temp_warranty_df.iterrows():
                merged_row = pd.concat([row, warranty_row])

        df_list.append(merged_row)

    merged_df = pd.concat(df_list, axis=1).T
    return merged_df


def derive_temporal_data_features(merged_df):
    merged_df.sort_values(['consultationid', 'timestamp'], inplace=True)

    merged_df['year'] = merged_df['timestamp'].dt.year
    merged_df['month'] = merged_df['timestamp'].dt.month
    merged_df['dayOfWeek'] = merged_df['timestamp'].dt.dayofweek
    merged_df['weekOfYear'] = merged_df['timestamp'].dt.isocalendar().week
    merged_df['timeSinceLastActivitySec'] = (merged_df.groupby('consultationid')['timestamp'].diff().dt.total_seconds()).fillna(0)
    merged_df['elapsedTimeSec'] = merged_df.groupby('consultationid')['timestamp'].transform(lambda x: (x - x.min())).dt.total_seconds()

    def month_to_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
            
    merged_df['season'] = merged_df['month'].apply(month_to_season)

    merged_df['builddate'] = pd.to_datetime(merged_df['builddate'], format='%Y-%m-%d').dt.tz_localize('UTC')
    merged_df['warrantydate'] = pd.to_datetime(merged_df['warrantydate'], format='%Y-%m-%d').dt.tz_localize('UTC')
    merged_df['vehicleAgeAtSession'] = (merged_df['timestamp'] - merged_df['builddate']).dt.days / 365
    merged_df['daysSinceWarrantyStart'] = (merged_df['timestamp'] - merged_df['warrantydate']).dt.days

    merged_df.drop(columns=['builddate', 'warrantydate', 'dtcdescription', 'v_warr_date_event', 'i_p_css_description',
                            'i_original_ccc_description', 'i_cpsc_description', 'i_css_description', 'ic_customer_verbatim',
                            'ic_technical_verbatim', 'i_incident_date', 'ic_accepted_date', 'warranty_anonymised_vin'],
                   inplace=True)
    return merged_df


def remove_outlier_diagnostic_activities(merged_df):
    activity_commonality = merged_df.value_counts('otxsequence')/merged_df['otxsequence'].count()
    activity_commonality = activity_commonality.reset_index()
    activity_commonality.columns = ['otxsequence', 'commonalityScore']

    mean = activity_commonality.commonalityScore.mean()
    std = activity_commonality.commonalityScore.std()
    print(f'MEAN: {mean}  STD: {std}')

    lower = mean - (2 * std)
    upper = mean + (2 * std)

    # Identify the outliers by checking for commonality score less than or greater than
    # lower and upper bounds respectively.
    outliers_condition = (activity_commonality.commonalityScore < lower) | (upper < activity_commonality.commonalityScore)
    most_common_activities = activity_commonality[outliers_condition]

    print(f"Most common activities with their commonality score (activities to be removed):\n{most_common_activities}")
    
    # Remove identified outlier (the most common) activities
    num_records_initial = len(merged_df)
    merged_df = merged_df[~merged_df.otxsequence.isin(most_common_activities.otxsequence)]

    print(f'Number of records removed: {num_records_initial - len(merged_df)}')
    return merged_df


def remove_duplicates(merged_df):
    num_records_initial = len(merged_df)
    merged_df.drop_duplicates()

    print(f'Number of duplicate records removed: {num_records_initial - len(merged_df)}')
    return merged_df


def handle_missing_vals(merged_df):
    # Apply 'Unknown' category to all missing values on categorical variables
    unordered_cat_cols = ['anonymised_vin', 'consultationid', 'otxsequence', 'model', 'modelyear', 'driver',
                          'plant', 'engine', 'transmission', 'module', 'dtcbase', 'faulttype', 'dtcfull',
                          'softwarepartnumber', 'hardwarepartnumber', 'i_p_css_code', 'i_original_ccc_code', 
                          'i_original_vfg_code','i_original_function_code', 'i_original_vrt_code', 'i_current_vfg_code',
                          'i_current_function_code', 'i_current_vrt_code',	'i_cpsc_code', 'i_cpsc_vfg_code',
                          'i_css_code', 'v_transmission_code','v_drive_code', 'v_engine_code', 'ic_repair_dealer_id',
                          'ic_eng_part_number', 'ic_serv_part_number','ic_part_suffix', 'ic_part_base', 'ic_part_prefix', 
                          'ic_causal_part_id', 'ic_repair_country_code', 'ic_replaced_yn']

    for col in unordered_cat_cols:
        merged_df[col] = merged_df[col].fillna('Unknown')

    # If daysSinceWarrantyStart NaN, then warrantydate was empty, meaning it is likely that warranty 
    # has not started on the vehicle
    merged_df['daysSinceWarrantyStart'].fillna(0, inplace=True)
    
    # If i_mileage NaN, then use current odometer mileage reading (odomiles)
    merged_df['i_mileage'].fillna(merged_df['odomiles'], inplace=True)

    # Fill NaN values in 'i_months_in_service' with the values from 'daysSinceWarrantyStart' turned into months
    # assuming that the average month is 30.44 days
    merged_df['i_months_in_service'] = merged_df.apply(
        lambda row: 0 if row['daysSinceWarrantyStart'] == 0 else ((row['daysSinceWarrantyStart'] / 30.44) if pd.isnull(row['i_months_in_service']) and pd.notnull(row['daysSinceWarrantyStart']) else row['i_months_in_service']),
        axis=1
    )
    merged_df['i_months_in_service'] = merged_df['i_months_in_service'].apply(lambda x: round(x) if pd.notnull(x) else x)

    # Fill NaN values in 'i_time_in_service' with the values from 'i_months_in_service' + 1
    # as this seems to be the pattern in the data
    merged_df['i_time_in_service'].fillna(merged_df['i_months_in_service'] + 1, inplace=True)
    
    na_counts = merged_df.isna().sum()
    na_columns = na_counts[na_counts > 0]

    if len(na_columns) > 0:
        print(f"Data fields with NaN values:\n{na_columns}")
        print(f'Total number of records in the DataFrame: {len(merged_df)}')
    else:
        print('There are no missing values in the DataFrame.')
    
    return merged_df


def standardise_num_data(merged_df):
    float_cols = ['elapsedTimeSec', 'timeSinceLastActivitySec', 'odomiles', 'vehicleAgeAtSession',
              'daysSinceWarrantyStart', 'i_mileage', 'i_time_in_service', 'i_months_in_service']
    for col in float_cols:
        merged_df[col] = merged_df[col].astype('float64')
    
    data_scaler = StandardScaler()
    merged_df[float_cols] = data_scaler.fit_transform(merged_df[float_cols])
    return merged_df


def save_csv(df, filename, append=False):
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    if append:
        df.to_csv(filename, mode='a', header=False, index=False) # Append to existing CSV
    else:
        df.to_csv(filename, index=False)


def process_data_for_training(filename):
    diagnostic_df = load_data(filename)
    print(f'{filename} - Data import complete\n')
    
    diagnostic_df = initiate_diagnostic_consultation(diagnostic_df)
    print(f'{filename} - Diagnostic consultation initiated\n')
    
    diagnostic_df = derive_vehicle_state_data(diagnostic_df)
    print(f'{filename} - Vehicle state data derived\n')
    
    warranty_df = pd.read_csv(f'data/claims_all.csv', low_memory=False)
    merged_df = merge_diag_warr_data(diagnostic_df, warranty_df)
    print(f'{filename} - Diagnostic and warranty data merged\n')
    
    merged_df = derive_temporal_data_features(merged_df)
    print(f'{filename} - Temporal features derived\n')
    
    merged_df = handle_missing_vals(merged_df)
    print(f'{filename} - Missing values addressed\n')
    
    merged_df = remove_duplicates(merged_df)
    print(f'{filename} - Duplicates removed\n')
    
    merged_df = standardise_num_data(merged_df)
    print(f'{filename} - Numerical data normalised\n')
    
    return merged_df


def append_no_warr_data(diagnostic_df):
    warranty_df = pd.DataFrame(columns=['warranty_anonymised_vin', 'i_incident_date', 'i_p_css_code', 'i_mileage', 'i_original_ccc_code',
                                        'i_time_in_service', 'i_months_in_service', 'i_original_vfg_code',
                                        'i_original_function_code',	'i_original_vrt_code', 'i_current_vfg_code',
                                        'i_current_function_code', 'i_current_vrt_code', 'i_cpsc_code', 'i_cpsc_vfg_code',
                                        'i_css_code', 'v_transmission_code', 'v_drive_code', 'v_engine_code',
                                        'v_warr_date_event', 'ic_repair_dealer_id', 'ic_repair_country_code',
                                        'ic_causal_part_id', 'ic_part_prefix', 'ic_part_base', 'ic_part_suffix',
                                        'ic_eng_part_number', 'ic_serv_part_number', 'ic_replaced_yn', 'ic_accepted_date',
                                        'i_p_css_description', 'i_original_ccc_description', 'i_cpsc_description', 'i_css_description',
                                        'ic_customer_verbatim', 'ic_technical_verbatim'])

    df_list = []
    for idx, row in diagnostic_df.iterrows():
        merged_row = row.copy()
        merged_row = pd.concat([merged_row, pd.Series([np.nan]*len(warranty_df.columns), index=warranty_df.columns)], axis=0)
        df_list.append(merged_row)

    merged_df = pd.concat(df_list, axis=1).T
    return merged_df


def process_data_for_predictions(filename):
    diagnostic_df = load_data(filename)
    diagnostic_df = initiate_diagnostic_consultation(diagnostic_df)
    diagnostic_df = derive_vehicle_state_data(diagnostic_df)
    merged_df = append_no_warr_data(diagnostic_df)
    merged_df = derive_temporal_data_features(merged_df)
    merged_df = handle_missing_vals(merged_df)
    merged_df = remove_duplicates(merged_df)
    merged_df = standardise_num_data(merged_df)

    merged_df = merged_df.drop(columns=['sessiontimestamp', 'timestamp'])
    return merged_df




