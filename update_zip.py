import argparse
import zipfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', type=str, required=True)
    parser.add_argument('--file', type=str, required=True)
    return parser.parse_args()


def update_zip():
    args = parse_args()
    with zipfile.ZipFile(args.zip, 'a') as zipf:
        source_path = args.file
        destination = 'descriptors.pkl'
        zipf.write(source_path, destination)


if __name__ == '__main__':
    update_zip()