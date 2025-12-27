from parsers import org
from embedding import embed_df
from config import config
from server import app
import uvicorn


def load():
    df = org.org_files_to_dataframes()
    embed_df(df)


def main():
    uvicorn.run(app, host="0.0.0.0", port=config["port"])


if __name__ == "__main__":
    main()
