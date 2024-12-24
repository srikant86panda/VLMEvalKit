import argparse
import hashlib
import os
import os.path as osp

def md5(s):
    hash = hashlib.new('md5')
    if osp.exists(s):
        with open(s, 'rb') as f:
            for chunk in iter(lambda: f.read(2**20), b''):
                hash.update(chunk)
    else:
        hash.update(s.encode('utf-8'))
    return str(hash.hexdigest())

def process_file(file_path):
    return {file_path: md5(file_path)}

def process_directory(directory):
    results = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tsv'):
                file_path = osp.join(root, file)
                results[file_path] = md5(file_path)
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate MD5 hashes for a string, file, or all .csv files in a directory.")
    parser.add_argument('input', type=str, help="String, file path, or directory path to hash.")

    args = parser.parse_args()
    input_path = args.input
    if osp.isfile(input_path):
        result = process_file(input_path)
        for path, hash_value in result.items():
            print(f"{path}: {hash_value}")
    elif osp.isdir(input_path):
        results = process_directory(input_path)
        print(f'results: {results}')
    else:
        print(f"Invalid input: {input_path} is neither a valid file nor a directory.")

if __name__ == "__main__":
    main()