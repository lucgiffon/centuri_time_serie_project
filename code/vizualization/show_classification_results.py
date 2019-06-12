import pandas as pd
import pathlib

if __name__ == "__main__":
    output_dir = pathlib.Path("/home/luc/PycharmProjects/centuri_project1/results/classification/cnn")
    p = output_dir.glob('**/*')
    files = [x for x in p if x.is_file()]

    lst_df = []
    for f in files:
        lst_df.append(pd.read_csv(f))

    bigdf = pd.concat(lst_df)

    bigdf.to_csv(output_dir / "allresults.csv")