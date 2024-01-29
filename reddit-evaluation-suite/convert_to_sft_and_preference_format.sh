python convert_to_sft_and_preference_format.py\
    --submissions_file_pattern s3://ai2-llm/pretraining-data/sources/reddit/science_subreddits/submissions_merged_*.jsonl.gz\
    --comments_file_pattern s3://ai2-llm/pretraining-data/sources/reddit/science_subreddits/comments_merged_*.jsonl.gz\
    --output_pref_file /net/nfs.cirrascale/allennlp/sachink/community-lm/data/science/v2/preference_data.jsonl\
    --output_sft_file /net/nfs.cirrascale/allennlp/sachink/community-lm/data/science/v2/sft_data.jsonl\
    --experiment science_subreddits\
    --submissions_tagger_name science_subreddit_submissions\
    --comments_tagger_name science_subreddit_comments