python -u train.py --batch_size=16 --data_dir='/data/PublicDataSets/SyntheticTextInWild/' --train_list='/data/PublicDataSets/SyntheticTextInWild/trainlist.txt' --valid_list='/data/PublicDataSets/SyntheticTextInWild/validation_list.txt' --debug=0 2>&1 | tee log/train.log