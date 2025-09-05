import os
from app import Me


def main():
    os.environ["REBUILD_KNOWLEDGE"] = "1"
    _ = Me()
    print("Knowledge index rebuilt.")


if __name__ == "__main__":
    main()
