from dotenv import load_dotenv

def main():
    load_dotenv()  # Loads variables from .env into os.environ
    print("Hello from org-sage!")


if __name__ == "__main__":
    main()
