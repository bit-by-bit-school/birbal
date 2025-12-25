from dotenv import load_dotenv
from parse_org_roam import org_files_to_dataframes

def main():
    load_dotenv()  # Loads variables from .env into os.environ
    df = org_files_to_dataframes()
    print(df["text_to_encode"])


if __name__ == "__main__":
    main()
