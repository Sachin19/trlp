python convert_to_preference_format.py\
    --submissions_file_pattern s3://ai2-llm/pretraining-data/sources/reddit/science_subreddits/submissions-*.json.gz\
    --comments_file_pattern s3://ai2-llm/pretraining-data/sources/reddit/science_subreddits/comments-*.json.gz\
    --output_file /net/nfs.cirrascale/allennlp/sachink/community-lm/reddit-evaluation-suite/science/science_preference_data.jsonl