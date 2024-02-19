factor=$1
dest=s3://ai2-llm/pretraining-data/sources/reddit/${factor}
mkdir -p $dest
echo $dest
dolma tag \
    --experiment ${factor}_subreddits \
    --destination $dest\
    --documents \
        's3://ai2-llm/pretraining-data/sources/reddit/v0/comments_merged_*.jsonl.gz' \
    --taggers ${factor}_subreddit_comments \
    --tagger_modules /net/nfs.cirrascale/allennlp/sachink/community-lm/trlp/reddit-evaluation-suite/${factor}_comments_tagger.py\
    --processes 32