import pandas as pd
import sys
import pickle
pickle.HIGHEST_PROTOCOL = 4
## Turn the pickle into a pandas hdf object.
def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]
    keyfile = sys.argv[3]

    df_dict = pd.read_pickle(infile)
    for k,df in df_dict.items():
        df = df.apply(lambda row: row*row['read_num'], axis=1)
        df = df.fillna(0)
        df = df.drop(["read_num", "kd_allele", "kd_conf", "hist_peak"], errors='ignore', axis=1).astype(int)
        df.to_hdf(outfile, "data/{}".format(k))

    with open(keyfile, 'w') as f:
        for k in df_dict.keys():
            f.write("data/{}\n".format(k))
if __name__ == "__main__":
   main()
