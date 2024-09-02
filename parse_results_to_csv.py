import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="Directory where result files are stored")
    parser.add_argument("--extensions", type=str, nargs="*", help="Extensions for saved result files")
    parser.add_argument(
        "--save_file_name", type=str, help="Name of file where all the concatenated results are stored"
    )
    args = parser.parse_args()
    return args


def parse_single_file_to_dict(filename: str) -> List[Dict[str, Any]]:
    list_of_dict = []
    with open(filename, mode="r") as f:
        lines = f.readlines()
        parse_string = ""
        for line in lines:
            l = line.strip()
            l = re.sub(": null", ": None", l)
            l = re.sub(": true", ": True", l)
            l = re.sub(": false", ": False", l)
            if l == "}{":
                parse_string += "}"
                parsed_dict = eval(parse_string)
                list_of_dict.append(parsed_dict)
                parse_string = "{"
            else:
                parse_string += l
        if parse_string:
            parsed_dict = eval(parse_string)
            list_of_dict.append(parsed_dict)
    return list_of_dict


def get_df_from_list_of_dict(list_of_dict: List[Dict[str, Any]]) -> pd.DataFrame:
    unpacked_results = []
    for record in tqdm(list_of_dict):
        result = {}
        for key, value in record.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    result[k] = v
            else:
                result[key] = value
        unpacked_results.append(result)

    return pd.DataFrame.from_records(unpacked_results).drop_duplicates()


def main():
    args = get_args()
    assert args.save_file_name.endswith(".csv"), "save_file_name must end with .csv"

    path = Path(args.output_dir).absolute()
    all_result_files = [p.__str__() for p in path.rglob("*") if p.__str__().split(".")[-1] in args.extensions]
    print(f"Grab {len(all_result_files)} files to parse")

    parsed_list_of_dict = []
    for filename in tqdm(all_result_files, desc="parsing results"):
        parsed_list_of_dict.extend(parse_single_file_to_dict(filename))

    df_results: pd.DataFrame = get_df_from_list_of_dict(parsed_list_of_dict)
    df_results.to_csv(args.save_file_name, index=False)


if __name__ == "__main__":
    main()
