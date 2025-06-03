import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':
    # Read the ngram stats
    ngrams = {
        1: pd.read_parquet('/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/df_n_gram_counts_1.parquet'),
        2: pd.read_parquet('/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/df_n_gram_counts_2.parquet'),
        3: pd.read_parquet('/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/df_n_gram_counts_3.parquet'),
        4: pd.read_parquet('/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/df_n_gram_counts_4.parquet'),
    }
    
    results = []

    for n, df_ngram in tqdm(ngrams.items()):
        grouped_stats = df_ngram.groupby('pid')['count'].agg(['mean', 'median', 'min', 'max']).reset_index()
        grouped_stats['n'] = n
        results.append(grouped_stats)

    results_df = pd.concat(results, ignore_index=True)

    OUTPUT_FILE = '/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/ngram_stats.csv'
    results_df.to_csv(OUTPUT_FILE, index=False)
    print("Ngram stats saved to: ", OUTPUT_FILE)
