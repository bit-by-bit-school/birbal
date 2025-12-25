from dotenv import load_dotenv
from parse_org_roam import org_files_to_dataframes
from embedding import embed_df

def main():
    load_dotenv()
    df = org_files_to_dataframes()
    embed_df(df)


if __name__ == "__main__":
    main()
