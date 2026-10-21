"""
@Author   : Xin Wu
@Time     : 2026/3/10 10:48
@File     : Search_data.py
@Remark   : 
"""


import pandas as pd
from typing import Optional


class BinaryKeyDataset:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, header=None)

        df["key"] = df.iloc[:, :13].astype(str).agg("".join, axis=1)
        df["value"] = df.iloc[:, 13]

        self.lookup = dict(zip(df["key"], df["value"]))
        print(f"There are {len(self.lookup)} entries in the dataset.")

    def get(self, key: str) -> Optional[float]:
        return self.lookup.get(key)

    def __getitem__(self, key: str) -> float:
        result = self.lookup.get(key)
        if result is None:
            raise KeyError(f"Key '{key}' not found in the dataset.")
        return result

    def __contains__(self, key: str) -> bool:
        return key in self.lookup

    def __len__(self) -> int:
        return len(self.lookup)


def main():
    csv_path = "./dataset.csv"
    dataset = BinaryKeyDataset(csv_path)

    print("Enter a 13-bit binary key to query (or 'q' to quit):")
    while True:
        key = input("> ").strip()
        if key.lower() == "q":
            print("886~")
            break
        elif len(key) != 13 or not all(c in "01" for c in key):
            print("Invalid key. Please enter a 13-bit binary string (e.g. 0100001101011).")
        elif key in dataset:
            print(f"Value: {dataset[key]}")
        else:
            print(f"Key '{key}' not found in the dataset.")


if __name__ == "__main__":
    main()