from dotenv import load_dotenv
from parse_org_roam import org_files_to_dataframes
from embedding import embed_df
from server import app
import uvicorn

def load():
    df = org_files_to_dataframes()
    embed_df(df)

def main():
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()
