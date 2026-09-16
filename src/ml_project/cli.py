import argparse

from ml_project.predict import predict
from ml_project.train import train


def main() -> None:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    train_parser = subparsers.add_parser("train")
    predict_parser = subparsers.add_parser("predict")

    train_parser.set_defaults(func=train)
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func()


if __name__ == "__main__":
    main()
