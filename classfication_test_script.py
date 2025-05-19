import os
import pandas as pd
import numpy as np
import joblib

def MergeDatasets(demos, dlcs, gamalytic, info, OutputFileName):
    print("Starting dataset merge...")
    for df in [demos, dlcs, gamalytic, info]:
        df.columns = df.columns.str.strip()
    demos.rename(columns={"full_game_appid": "Full_game_appid"}, inplace=True)
    dlcs.rename(columns={"base_appid": "Full_game_appid"}, inplace=True)
    gamalytic.rename(columns={"steamId": "Full_game_appid"}, inplace=True)
    info.rename(columns={"appid": "Full_game_appid"}, inplace=True)

    for df in [demos, dlcs, gamalytic, info]:
        df["Full_game_appid"] = df["Full_game_appid"].astype(str)

    merged_df = demos.merge(dlcs, on="Full_game_appid", how="left") \
                     .merge(gamalytic, on="Full_game_appid", how="right") \
                     .merge(info, on="Full_game_appid", how="left")

    merged_df.drop(columns=['Unnamed: 0', 'demo_appid', 'dlc_appid', 'Full_game_appid','name'], inplace=True, errors='ignore')

    output_folder = r"C:\Users\Admin\Downloads\ML milestone 2\data\raw"
    output_file = f"{OutputFileName}.csv"
    output_path = os.path.join(output_folder, output_file)
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged dataset to: {output_path}")
    print("Merged shape:", merged_df.shape)
    print("Dataset merge completed.\n")
    return merged_df

def preprocess_classification_dataset(df, median_values, mode_values, numerical_means,
                                      label_encoder_publisherClass, label_encoder_reviewScore,
                                      standard_scaler, bool_features_processed):
    print("Starting preprocessing for classification...")
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col, median in median_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(median)
    print("Filled missing numerical values with median values.")

    for col, mode_val in mode_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)
    print("Filled categorical columns with mode values.")

    if 'name' in df.columns:
        df.drop(columns='name',inplace=True)

    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        mean_year = numerical_means.get('year', 2000)
        mean_month = numerical_means.get('month', 6)
        mean_day = numerical_means.get('day', 15)
        df['year'] = df['release_date'].dt.year.fillna(mean_year)
        df['month'] = df['release_date'].dt.month.fillna(mean_month)
        df['day'] = df['release_date'].dt.day.fillna(mean_day)
        current_year = 2025
        df['age_years'] = current_year - df['year']
        df.drop(columns='release_date', inplace=True)
        print("Processed release dates into year, month, day and age features.")

    for feature in bool_features_processed:
        if feature in df.columns:
            df[feature] = df[feature].fillna(False).astype(int)
    print("Converted boolean features to integers.")

    if 'publisherClass' in df.columns:
        known_publishers = set(label_encoder_publisherClass.classes_)
        df['publisherClass'] = df['publisherClass'].apply(
            lambda x: x if pd.notna(x) and x in known_publishers else 'Unknown'
        )
        if 'Unknown' not in known_publishers:
            all_publishers = list(known_publishers) + ['Unknown']
            new_encoder = joblib.load('label_encoder_publisherClass.pkl')  
            df['publisher_encoded'] = new_encoder.transform(df['publisherClass'])
        else:
            df['publisher_encoded'] = label_encoder_publisherClass.transform(df['publisherClass'])
        df.drop(columns='publisherClass', inplace=True)
        print("Encoded publisher classes.")

    if 'genres' in df.columns:
        df['genres'] = df['genres'].fillna('')
        genres = df['genres'].str.split(',', expand=True)
        unique_genres = pd.unique(genres.values.ravel())
        unique_genres = [g for g in unique_genres if pd.notna(g)]
        for genre in unique_genres:
            genre_name = genre.strip()
            df[genre_name] = df['genres'].str.contains(genre, regex=False).astype(int)
        df['total_genres'] = df['genres'].str.count(',') + (df['genres'] != '').astype(int)
        df.drop(columns='genres', inplace=True)
        print(f"Processed genres into {len(unique_genres)} one-hot encoded columns.")

    if 'supported_platforms' in df.columns:
        df['supported_platforms'] = df['supported_platforms'].fillna('')
        df['windows'] = df['supported_platforms'].str.contains('windows', case=False).astype(int)
        df['linux'] = df['supported_platforms'].str.contains('linux', case=False).astype(int)
        df['mac'] = df['supported_platforms'].str.contains('mac', case=False).astype(int)
        df['total_platforms'] = df[['windows', 'linux', 'mac']].sum(axis=1)
        df.drop(columns='supported_platforms', inplace=True)
        print("Processed platform information.")

    if 'reviewScore' in df.columns:
        df['reviewScore'] = label_encoder_reviewScore.transform(df['reviewScore'])
        print(f"Encoded reviewScore classes. Classes: {label_encoder_reviewScore.classes_}")

    numerical_features = standard_scaler.feature_names_in_
    numerical_cols = [col for col in numerical_features if col in df.columns]
    if numerical_cols:
        df[numerical_cols] = standard_scaler.transform(df[numerical_cols])
        print(f"Applied standard scaling to {len(numerical_cols)} numerical features.")

    print("Classification preprocessing completed.\n")
    return df

def main():
    try:
        print("Loading datasets...")
        demos = pd.read_csv(r"C:\Users\Admin\Downloads\ML milestone 2\data\raw\demos.csv")
        dlcs = pd.read_csv(r"C:\Users\Admin\Downloads\ML milestone 2\data\raw\dlcs.csv")
        gamalytic = pd.read_csv(r"C:\Users\Admin\Downloads\ML milestone 2\data\raw\ms2_gamalytic_steam_games.csv")
        info = pd.read_csv(r"C:\Users\Admin\Downloads\ML milestone 2\data\raw\info_base_games.csv")
        print("Datasets loaded.")

        print("Merging datasets...")
        merged_df = MergeDatasets(demos, dlcs, gamalytic, info, "classification_merged_dataset")

        print("Loading classification preprocessing artifacts...")
        median_values = joblib.load('median_values.joblib')
        mode_values = joblib.load('mode_values.joblib')
        numerical_means = joblib.load('numerical_means.joblib')
        label_encoder_publisherClass = joblib.load('label_encoder_publisherClass.pkl')
        label_encoder_reviewScore = joblib.load('label_encoder_reviewScore.pkl')
        standard_scaler = joblib.load('standard_scaler.pkl')
        
        rf_classifier = joblib.load('best_rf_model.joblib')
        gb_classifier = joblib.load('best_rf_model.joblib')
        print("All classification preprocessing artifacts loaded successfully.\n")

        expected_features = gb_classifier.feature_names_in_ if hasattr(gb_classifier, 'feature_names_in_') else [col for col in merged_df.columns if col != 'reviewScore']

        print("Preprocessing dataset for classification...")
        processed_df = preprocess_classification_dataset(
            merged_df,
            median_values=median_values,
            mode_values=mode_values,
            numerical_means=numerical_means,
            label_encoder_publisherClass=label_encoder_publisherClass,
            label_encoder_reviewScore=label_encoder_reviewScore,
            standard_scaler=standard_scaler,
            bool_features_processed=['steam_achievements', 'steam_trading_cards', 'workshop_support']
        )

        if 'reviewScore' in processed_df.columns:
            y_true = processed_df.pop('reviewScore')
            print(f"Extracted target variable with {len(y_true)} values.")

        if hasattr(rf_classifier, 'feature_names_in_'):
            print("Aligning features to exactly match model's expected features...")
            final_df = pd.DataFrame()
            for feature in expected_features:
                final_df[feature] = processed_df[feature] if feature in processed_df.columns else 0
            processed_df = final_df

        print("Running classification prediction...")
        predictions_proba = rf_classifier.predict_proba(processed_df)
        predictions = rf_classifier.predict(processed_df)

        results_df = pd.DataFrame({
            'Predicted_ReviewScore_Encoded': predictions
        })

        
        for i, class_name in enumerate(label_encoder_reviewScore.classes_):
            if i < predictions_proba.shape[1]:
                results_df[f'Probability_{class_name}'] = predictions_proba[:, i]
            else:
                results_df[f'Probability_{class_name}'] = 0.0

        results_df['Predicted_ReviewScore_Class'] = results_df['Predicted_ReviewScore_Encoded'].apply(
            lambda x: label_encoder_reviewScore.inverse_transform([x])[0]
        )

        # Combine original merged_df with predictions on the left
        combined_df = pd.concat([merged_df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

        output_path = "classification_predictions.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"Classification predictions with original data saved to {output_path}")

        if 'y_true' in locals():
            from sklearn.metrics import accuracy_score, classification_report

            # Fix dtype mismatch before evaluation
            if y_true.dtype == 'O':
                try:
                    y_true = y_true.astype(int)
                    print("Converted y_true to int type for evaluation.")
                except Exception as e:
                    print(f"Failed to convert y_true to int: {e}")
            else:
                y_true = y_true.astype(predictions.dtype)

            accuracy = accuracy_score(y_true, predictions)
            print(f"Model accuracy on this dataset: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_true, predictions))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
