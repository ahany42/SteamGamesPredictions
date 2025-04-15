import pandas as pd
def combine_datasets():
    demos = pd.read_csv("datasets/raw/demos.csv")
    dlcs = pd.read_csv("datasets/raw/dlcs.csv")
    gamalytic = pd.read_csv("datasets/raw/gamalytic_steam_games.csv")
    info = pd.read_csv("datasets/raw/info_base_games.csv", low_memory=False)

    demos.columns = demos.columns.str.strip()
    dlcs.columns = dlcs.columns.str.strip()
    gamalytic.columns = gamalytic.columns.str.strip()
    info.columns = info.columns.str.strip()

    demos.rename(columns={"full_game_appid": "Full_game_appid"}, inplace=True)
    dlcs.rename(columns={"base_appid": "Full_game_appid"}, inplace=True)
    gamalytic.rename(columns={"steamId": "Full_game_appid"}, inplace=True)
    info.rename(columns={"appid": "Full_game_appid"}, inplace=True)

    demos["Full_game_appid"] = demos["Full_game_appid"].astype(str)
    dlcs["Full_game_appid"] = dlcs["Full_game_appid"].astype(str)
    gamalytic["Full_game_appid"] = gamalytic["Full_game_appid"].astype(str)
    info["Full_game_appid"] = info["Full_game_appid"].astype(str)

    merged_df = demos.merge(dlcs, on="Full_game_appid", how="outer") \
                    .merge(gamalytic, on="Full_game_appid", how="outer") \
                    .merge(info, on="Full_game_appid", how="outer")

    merged_df.to_csv("datasets/raw/combined_games.csv", index=False)
    print("Demos:", demos.shape)
    print("DLCs:", dlcs.shape)
    print("Gamalytic:", gamalytic.shape)
    print("Merged shape:",merged_df.shape)
    if 'reviewScore' in merged_df.columns:
        review_score = merged_df.pop('reviewScore')