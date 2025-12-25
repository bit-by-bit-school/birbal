from dotenv import load_dotenv
from parse_org_roam import org_files_to_dataframes
from embedding import embed_df
from query import query_vector

def load():
    df = org_files_to_dataframes()
    embed_df(df)

def main():
    load_dotenv()
    print(query_vector("What are examples of search sort correspondence?"))



if __name__ == "__main__":
    main()
