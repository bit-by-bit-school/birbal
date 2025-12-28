from birbal.parsers import *
from birbal.embedding import embed_df


def get_all_filenames_in_file_dir(extension):
    org_path = config["file_dir"]
    path = os.path.join(org_path, f"**/*.{extension}")
    files = glob.glob(path, recursive=True)

    return files


def files_to_dataframe(files):
    parser = OrgParser
    accumulated_df = pd.concat([parser.parse(file) for file in files])
    return accumulated_df


def ingest():
    files = get_all_filenames_in_file_dir("org")
    df = files_to_dataframe(files)
    embed_df(df)


if __name__ == "__main__":
    ingest()
