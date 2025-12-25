from dotenv import load_dotenv
from parse_org_roam import org_files_to_dataframes
from embedding import embed_df
from llm import query_llm

def load():
    df = org_files_to_dataframes()
    embed_df(df)

def main():
    load_dotenv()
    query_llm("What are all the things I need to do to run Org Roam UI on my ipad?")


if __name__ == "__main__":
    main()
