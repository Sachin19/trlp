factor=$1
python convert_to_sft_and_preference_format.py\
    --submissions_file_pattern s3://ai2-llm/pretraining-data/sources/reddit/$factor/submissions_merged_*.jsonl.gz\
    --comments_file_pattern s3://ai2-llm/pretraining-data/sources/reddit/$factor/comments_merged_*.jsonl.gz\
    --output_pref_file /net/nfs.cirrascale/allennlp/sachink/community-lm/data/$factor/preference_data.jsonl\
    --output_sft_file /net/nfs.cirrascale/allennlp/sachink/community-lm/data/$factor/sft_data.jsonl\
    --experiment ${factor}_subreddits\
    --submissions_tagger_name ${factor}_subreddit_submissions\
    --comments_tagger_name ${factor}_subreddit_comments