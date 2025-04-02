import pickle

abbreviate = {
    "wikifact": "wiki",
    "synthetic_reasoning": "srea",
    "truthful_qa": "truth",
    "math": "math",
    "gsm": "gsm",
    "babi_qa": "babi",
    "bbq": "bbq",
    "thai_exam": "thai",
    "legal_support": "lsup",
    "legalbench": "lben",
    "civil_comments": "civ",
    "dyck_language_np=3": "dyck",
    "air_bench_2024": "air",
    "med_qa": "med",
    "raft": "rft",
    "mmlu": "mmlu",
    "entity_matching": "emat",
    "boolq": "bool",
    "entity_data_imputation": "eimp",
    "commonsense": "comm",
    "imdb": "imdb",
    "blimp": "blimp"
}

if __name__ == "__main__":
    with open("../data/results.pkl", "rb") as f:
        results = pickle.load(f)
        
    # Extract the sub-dataframe with the 16 desired columns.
    sub_df = results.sample(n=200, axis=1, random_state=42)

    # Replace -1 with an empty string.
    sub_df = sub_df.replace(-1, "NA")

    # Drop any rows where all values are empty.
    sub_df = sub_df.loc[~(sub_df.eq("NA").all(axis=1))]

    # Rename the columns with the new flat names.
    new_column_names = [f"{abbreviate[col[sub_df.columns.names.index('scenario')]]}" for col in sub_df.columns]
    sub_df.columns = new_column_names

    # Remove any column or index names.
    sub_df.columns.name = None
    sub_df.index.name = None

    # Save the resulting DataFrame to a CSV file without the row index.
    sub_df.to_csv("../data/deval_resmat.csv", index=False)

    total_cells = sub_df.size
    missing_cells = (sub_df == "NA").sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100

    # Print the percent of missing data.
    print("Percent missing data: {:.2f}%".format(missing_percentage))