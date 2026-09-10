import argparse
from pprint import pprint

from src.data_inspection import inspect_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Student Success Early-Signal & Planning System")
    parser.add_argument(
        "--inspect-data",
        action="store_true",
        help="Run Phase 1 data-package inspection.",
    )
    args = parser.parse_args()

    if args.inspect_data:
        print("Phase 1 data inspection")
        pprint(inspect_package())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
