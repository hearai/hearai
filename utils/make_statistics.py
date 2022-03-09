import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser

def main(file_path: str, output_directory: str, separator: str):
    df = pd.read_csv(file_path, sep=separator)

    # Prepare DataFrame to count values
    df = df.drop("name", axis="columns")
    df = df.drop("start", axis="columns")
    df = df.drop("end", axis="columns")
    df = df.dropna(axis="columns")
    
    for column_name in df.columns:
        file_name = column_name.replace(" ", "_").replace("/", "_")
        file_name = file_name+".png"
        file_path = os.path.join(output_directory, file_name)
        
        column_stats = df.loc[:, column_name].value_counts().sort_index()

        # Set figure size to plot a more data
        if len(column_stats.index) > 20:
            plt.figure(figsize=(15, 5))

        # Create and save plot
        plt.bar(column_stats.index, column_stats.values)
        plt.title(column_name)
        plt.xticks(column_stats.index)
        plt.savefig(file_path)

        # Clear figures
        plt.cla()
        plt.clf()


if __name__ == "__main__":
    parser = ArgumentParser(description="AA")

    parser.add_argument("--file_path", type=str, help="Path to file, which contains HamNoSys annotations")
    parser.add_argument("--output_directory", type=str, help="Path to the output directory")

    parser.add_argument("--separator", default=" ", help="Sign, which is used as a separator in the annotation file")
    args = parser.parse_args()
    
    main(args.file_path, args.output_directory, args.separator)