# News Article Matching

Code and Dataset for our paper **"Combining Lexical and Semantic Similarity Methods for News Article Matching"**

Dataset:

* `df_news_init.csv`: Turkish news dataset in `csv` format. `title`, `spot`, `body`, `lastModified`, `group_id` columns are used in the code. `group_id` is same for the matched news. There are also additional columns.
* `df_comparisons_init.csv`: Derived comparison pairs from `df_news_init.csv`. 1858 positive and 15000 negative pairs are obtained.

Notebooks:

* `v01_unsupervised.ipynb`: Unsupervised scoring methods without lemmatization and stop-words removel. Results in tha paper are at `v03.unsupervised_lemma.ipynb`.
* `v02_supervised.ipynb`: Supervised classification methods without lemmatization and stop-words removel. Results in the paper are at `v04.supervised_lemma.ipynb`.
* `v03_unsupervised_lemma.ipynb`: Unsupervised experiments.
* `v04_supevised_lemma.ipynb`: Supervised experiments.
* `v05_wmc.ipynb`: Words Mover's Distance experiments for comparison.
* `v06_supservised_we_concat.ipynb`: Word embedding concatentation as features to classifiers, did not work.
* `v07_kenter_short_text.ipynb`: BM25 similarity method experiments for comparison


