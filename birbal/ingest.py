from birbal.parsers.org import org_files_to_dataframes
from birbal.embedding import embed_df


def ingest():
    df = org_files_to_dataframes()
    embed_df(df)


if __name__ == "__main__":
    ingest()
