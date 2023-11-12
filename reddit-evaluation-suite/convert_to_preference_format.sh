python convert_to_preference_format.py\
    --submissions_file_pattern 'science/submissions_merged_*.jsonl.gz'\
    --comments_file_pattern s3://ai2-llm/pretraining-data/sources/reddit/science_subreddits/comments_merged_*.jsonl.gz\
    --output_file /net/nfs.cirrascale/allennlp/sachink/community-lm/science/science_preference_data.jsonl\
    --experiment science_subreddits\
    --submissions_tagger_name science_subreddit_submissions\
    --comments_tagger_name science_subreddit_comments