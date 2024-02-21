# export AWS_ACCESS_KEY_ID="ASIASHLPW4FESWKV2BRD"
# export AWS_SECRET_ACCESS_KEY="kV6rN3BTGFwa+Jl/CAuWqGgwNw22d/Cvkq9EF9pw"
# export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEJb//////////wEaCXVzLXdlc3QtMiJHMEUCICm+lneGeyBj6jmnxqANcESQ7hoPhYaMAfKiMR1UKKtLAiEA+/1AiS/FsMvypjAl65xqSuw65otnkI6TgmpHG4wxG4Mq9QIIz///////////ARAFGgwxNTMyNDI0OTMyNTciDMshugAHdNs8qU+ajyrJAjeiJsJn70L2oc+4z4SpmNJWhMd4vQk1ov7Xsd5wDa5hoNcCYi9aHK8SuCNzAmVwI2WgJumM79FKR+j1KEPnuTUxMljTUUFok6p7qwsiN5vV8QA9e25Xr8x1akcCeSiZRF6u1Rx+Ntx/1jD70/oPyHj0XQO2muvdnOHxkvcuMektyVvGI3iCxnGkpH8AhNKmIL5FZDvBkTtySPMYw4dqMrdTqvCSWDpQx74OYtFA0mI0c2CqjNFPTLAuVMiGuXeWa/zRQWS9hZy6otP5TQ44IJCanQ7sl7Iv6wGwK/p6fOP8Z0Aun5py+Rf05IKVzlkfwB3ZpBdJj7TThZpfqNEh/eYksURfOJdAQ6STtMFSL1c4NUwDdsqOxaFjoKE8DD7XioXuVpJdWJFpb7Rk5wwp2Y/XFRfgpIt2vvbMJ673VbCT9RxGi1fi6QxaMLThsaoGOqcBwJ7PinS1DlztpDT6WgPinr+Jbmin5r/bL4YbkO9fcEMU/H2GwbEDldSBYGdJiFbNiVkvWRkRxDzKWk+VQPnKqMsdGoyn44BtSj7xeAlGYHRbledcGPjBB/BvtTOMCNXIeiKVhZn2JAXuK5H+8wLwMe+I4JLgsQSh/aBDt3JhDpRDtA13xr3W5jumfPXQoJNsi03Qui0Bny3oX7di6/KU1NEptDud0w8="

# dolma tag \
#     --experiment science_subreddits \
#     --destination 's3://ai2-llm/pretraining-data/sources/reddit/science_subreddits/'\
#     --documents \
#         's3://ai2-llm/pretraining-data/sources/reddit/v0/submissions_merged_*.jsonl.gz' \
#     --taggers science_subreddit_submissions \
#     --tagger_modules /net/nfs.cirrascale/allennlp/sachink/community-lm/reddit-evaluation-suite/science_tagger.py\
#     --processes 16

mkdir -p s3://ai2-llm/pretraining-data/sources/reddit/science
dolma tag \
    --experiment science_subreddits \
    --destination 's3://ai2-llm/pretraining-data/sources/reddit/science'\
    --documents \
        's3://ai2-llm/pretraining-data/sources/reddit/v0/submissions_merged_*.jsonl.gz' \
    --taggers science_subreddit_submissions \
    --tagger_modules /net/nfs.cirrascale/allennlp/sachink/community-lm/trlp/reddit-evaluation-suite/science_tagger.py\
    --processes 16
    # /net/nfs.cirrascale/allennlp/sachink/community-lm/reddit-evaluation-suite/science/\