head -n 2500 data_raw_unsplit.rnk > valid.rnk
head -n 2500 data_raw_unsplit.ses > valid.ses
tail -n +2501 data_raw_unsplit.rnk > train.rnk
tail -n +2501 data_raw_unsplit.ses > train.ses

python convert-text2dict.py train train --cutoff=50000 --min_freq=5
python convert-text2dict.py valid valid --cutoff=50000 --min_freq=5 --dict=train.dict.pkl
