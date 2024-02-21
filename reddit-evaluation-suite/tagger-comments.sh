dolma tag \
    --experiment science_subreddits \
    --destination 's3://ai2-llm/pretraining-data/sources/reddit/science'\
    --documents \
        's3://ai2-llm/pretraining-data/sources/reddit/v0/comments_merged_*.jsonl.gz' \
    --taggers science_subreddit_comments \
    --tagger_modules /net/nfs.cirrascale/allennlp/sachink/community-lm/trlp/reddit-evaluation-suite/science_comments_tagger.py\
    --processes 32