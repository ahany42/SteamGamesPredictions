import os
import pandas as pd
import joblib
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
def MergeDatasets(demos, dlcs, gamalytic, info, OutputFileName):
    print("Merging datasets...")
    demos.columns = demos.columns.str.strip()
    dlcs.columns = dlcs.columns.str.strip()
    gamalytic.columns = gamalytic.columns.str.strip()
    info.columns = info.columns.str.strip()

    demos.rename(columns={"full_game_appid": "Full_game_appid"}, inplace=True)
    dlcs.rename(columns={"base_appid": "Full_game_appid"}, inplace=True)
    gamalytic.rename(columns={"steamId": "Full_game_appid"}, inplace=True)
    info.rename(columns={"appid": "Full_game_appid"}, inplace=True)

    for df in [demos, dlcs, gamalytic, info]:
        df["Full_game_appid"] = df["Full_game_appid"].astype(str)

    merged_df = demos.merge(dlcs, on="Full_game_appid", how="inner") \
                     .merge(gamalytic, on="Full_game_appid", how="right") \
                     .merge(info, on="Full_game_appid", how="inner")

    merged_df.drop(columns=['Unnamed: 0', 'demo_appid', 'dlc_appid', 'Full_game_appid'], inplace=True, errors='ignore')

    output_folder = r"C:\Users\Admin\Downloads\ML milestone 1\data\raw"
    output_file = f"{OutputFileName}.csv"
    output_path = os.path.join(output_folder, output_file)

    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged dataset to: {output_path}")
    print("Merged shape:", merged_df.shape)

    if 'reviewScore' in merged_df.columns:
        review_score = merged_df.pop('reviewScore')
        merged_df['reviewScore'] = review_score
    return merged_df

def preprocess_user_dataset(df, median_values, genresMode, dateFillValues,
                            label_encoder_PBClass, standard_scaler,
                            redundant_features, selected_features):
    print("Starting preprocessing...")
    df = df.fillna(median_values)
    df['genres'] = df['genres'].fillna(genresMode)
    df['name'] = df['name'].fillna('Unknown Game')
    df['name'] = df['name'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.lower().str.strip()
    df['name_tokens'] = df['name'].str.split()
    df['name_length'] = df['name'].apply(len)

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year.fillna(dateFillValues.get('year', 2000))
    df['month'] = df['release_date'].dt.month.fillna(dateFillValues.get('month', 1))
    df['day'] = df['release_date'].dt.day.fillna(dateFillValues.get('day', 1))
    df.drop(columns='release_date', inplace=True)

    bool_features = ['steam_achievements', 'steam_trading_cards', 'workshop_support']
    for feature in bool_features:
        if feature in df.columns:
            df[feature] = df[feature].astype(int)

    if 'publisherClass' in df.columns:
        df['publisher_class_encoded'] = label_encoder_PBClass.transform(df['publisherClass'])
        df.drop(columns='publisherClass', inplace=True)

    genres = df['genres'].str.split(',', expand=True)
    unique_genres = pd.unique(genres.values.ravel())
    for genre in unique_genres:
        if pd.isna(genre):
            continue
        genre_name = genre.strip()
        df[genre_name] = genres.apply(lambda row: 1 if genre_name in row.values else 0, axis=1)
    df.drop(columns='genres', inplace=True)

    platforms = df['supported_platforms'].str.split(',', expand=True)
    if platforms.shape[1] >= 3:
        df['windows'] = platforms[0].apply(lambda x: 1 if pd.notna(x) and 'windows' in x.lower() else 0)
        df['linux'] = platforms[1].apply(lambda x: 1 if pd.notna(x) and 'linux' in x.lower() else 0)
        df['mac'] = platforms[2].apply(lambda x: 1 if pd.notna(x) and 'mac' in x.lower() else 0)
    else:
        df['windows'] = df['linux'] = df['mac'] = 0
    df.drop(columns='supported_platforms', inplace=True)

    df['Total_Platforms'] = df[['windows', 'linux', 'mac']].sum(axis=1)

    # Calculate Total_Genres as sum of all genre columns created
    genre_columns = [g.strip() for g in unique_genres if pd.notna(g)]
    print("Genre columns detected:", genre_columns)
    df['Total_Genres'] = df[genre_columns].sum(axis=1)

    df.drop(columns=redundant_features, errors='ignore', inplace=True)

    # Scale numerical features
    X_to_scale = df[standard_scaler.feature_names_in_]
    X_scaled = standard_scaler.transform(X_to_scale)
    X_scaled_df = pd.DataFrame(X_scaled, columns=standard_scaler.feature_names_in_, index=df.index)
    df.update(X_scaled_df)

    # Fix columns with space prefixes if necessary
    if 'Indie' in df.columns and ' Indie' not in df.columns:
        df[' Indie'] = df['Indie']
        df.drop(columns=['Indie'], inplace=True)
    if 'Sports' in df.columns and ' Sports' not in df.columns:
        df[' Sports'] = df['Sports']
        df.drop(columns=['Sports'], inplace=True)

    # Add missing selected features as zeros
    missing_cols = set(selected_features) - set(df.columns)
    for col in missing_cols:
        print(f"Adding missing column '{col}' filled with zeros")
        df[col] = 0

    df = df[selected_features]

    print("Preprocessing complete. Final shape:", df.shape)
    print("Final features:", df.columns.tolist())
    return df

def main():
    try:
        print("Loading CSV files...")
        demos = pd.read_csv(r"C:\Users\Admin\Downloads\ML milestone 1\data\raw\demos.csv")
        dlcs = pd.read_csv(r"C:\Users\Admin\Downloads\ML milestone 1\data\raw\dlcs.csv")
        gamalyticMS1 = pd.read_csv(r"C:\Users\Admin\Downloads\ML milestone 1\data\raw\gamalytic_steam_games.csv")
        info = pd.read_csv(r"C:\Users\Admin\Downloads\ML milestone 1\data\raw\info_base_games.csv")

        merged_df = MergeDatasets(demos, dlcs, gamalyticMS1, info, "merged_dataset")

        print("Loading preprocessing objects...")
        genresMode = joblib.load('genres_mode.pkl')
        dateFillValues = joblib.load('date_fill_values.pkl')
        label_encoder_PBClass = joblib.load('label_encoder_PBClass.pkl')
        redundant_features = joblib.load('redundant_features.joblib')
        selected_features = ['Total_Genres', 'Action',' Indie','copiesSold', 'year', ' Sports']
        standard_scaler = joblib.load('standard_scaler.joblib')
        median_values = joblib.load('median_values.joblib')
        xgboost_model = joblib.load('xgb_model.joblib')

        print("Preprocessing merged data...")
        processed_df = preprocess_user_dataset(
            merged_df,
            median_values=median_values,
            genresMode=genresMode,
            dateFillValues=dateFillValues,
            label_encoder_PBClass=label_encoder_PBClass,
            standard_scaler=standard_scaler,
            redundant_features=redundant_features,
            selected_features=selected_features
        )

        expected_features = ['Total_Genres', 'copiesSold', ' Indie', ' Sports', 'year', 'Action']
        print("Checking for missing expected features before prediction...")
        missing_cols = [col for col in expected_features if col not in processed_df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns for prediction: {missing_cols}")

        input_for_model = processed_df[expected_features]
        print("Input DataFrame columns for prediction:", input_for_model.columns.tolist())

        print("Running prediction...")
        predictions = xgboost_model.predict(input_for_model)
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


        y_true = merged_df['reviewScore']
        y_pred = predictions

        
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

      
        print(f"\nModel Evaluation Metrics:")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE):{mae:.4f}")

        print("Saving merged dataset with predictions...")
        final_df = pd.concat([merged_df.reset_index(drop=True),
                              pd.Series(predictions, name='Predicted_ReviewScore')], axis=1)

        final_df.to_csv('regression_predictions.csv', index=False)
        print("Predictions saved to 'regression_predictions.csv'")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
