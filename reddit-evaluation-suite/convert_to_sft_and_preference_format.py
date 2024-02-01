import glob 

import argparse
import json

from datetime import datetime
from collections import defaultdict

import smart_open

import spacy
nlp = spacy.load("en_core_web_sm", exclude=["ner", "tok2vec"] )

from tqdm import tqdm

from constants import *

import boto3

s3_client = boto3.client('s3')

parser = argparse.ArgumentParser()
parser.add_argument("--submissions_file_pattern", required=True)
parser.add_argument("--comments_file_pattern", required=True)
parser.add_argument("--output_sft_file", required=True)
parser.add_argument("--output_pref_file", required=True)
parser.add_argument("--experiment", required=True)
parser.add_argument("--submissions_tagger_name", required=True)
parser.add_argument("--comments_tagger_name", required=True)
parser.add_argument("--max_comments_per_submissions", default=5, type=int)



args = parser.parse_args()

# protocol, submission_pattern = args.submissions_file_pattern.split("://")
# items = submission_pattern.split("/")
# bucket = items[0]
# prefix = "/".join(items[1:]).split("*")[0]
# print(protocol, bucket, prefix)

# client=boto3.client(protocol)
# submission_objs = client.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']
# # submissions_files = glob.glob(args.submissions_file_pattern)
# submission_files = [f"{protocol}://{bucket}/{obj['Key']}" for obj in submission_objs]
submission_files = glob.glob(args.submissions_file_pattern)

protocol, comment_pattern = args.comments_file_pattern.split("://")
items = comment_pattern.split("/")
bucket = items[0]
prefix = "/".join(items[1:]).split("*")[0]
print(protocol, bucket, prefix)

client=boto3.client(protocol)
comment_objs = client.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']
# comments_files = glob.glob(args.comments_file_pattern)
comments_files = [f"{protocol}://{bucket}/{obj['Key']}" for obj in comment_objs]
print(len(submission_files), len(comments_files))

#print(submission_files)
#print(comments_files)
## convert to (post, commentA, commentB setup)
comments_by_submission = {}

submission_prefix = f"{args.experiment}__{args.submissions_tagger_name}__"
comments_prefix = f"{args.experiment}__{args.comments_tagger_name}__"
# print(comments_prefix)
submission_prefix_len = len(submission_prefix)
comments_prefix_len = len(comments_prefix)
submissions_id2doc = {}

total_submissions = 0
for submissions_filename in submission_files:
    with smart_open.open(submissions_filename) as fsubmission:
        for submissiondoc in fsubmission:
            submissiondict = json.loads(submissiondoc)
            attributes = list(submissiondict['attributes'].keys())
            # print(attributes)
            submissions_id2doc[submissiondict['id']] = attributes[1][submission_prefix_len:] #metadata
            total_submissions += 1
            if total_submissions % 1000 == 0:
                print("\r")
                print(f"{total_submissions/1000}K", end="", flush=True)

total_comments = 0
# print(comments_files[:1])
for comments_filename in comments_files:
    with smart_open.open(comments_filename) as fcomment:
        for commentdoc in fcomment:
            total_comments += 1
            if total_comments % 1000 == 0:
                print("\r")
                print(f"{total_comments/1000}K", end="", flush=True)

            # print(comment_id)
            commentdict = json.loads(commentdoc)
            attributes = list(commentdict['attributes'].keys())
            # print(attributes[1])
            # input()
            commentmetadata = eval(attributes[1][comments_prefix_len:]) #metadata
            commenttext = attributes[0][comments_prefix_len:] #text
            # comment_time = datetime.fromisoformat(attributes[2][comments_prefix_len:]).timestamp() # text

            user_id_spans = commentmetadata['user_id_spans']
            if len(user_id_spans) > 1:
                endidx = user_id_spans[1][0]
            else:
                endidx = len(commenttext)
            
            commenttext = commenttext[user_id_spans[0][1]+2:endidx]
            submission_id = commentmetadata['thread_id']

            subreddit, submission_id, comment_id, commentscore = commentmetadata['subreddit'].lower(), commentmetadata['thread_id'], commentdict['id'], commentmetadata['comment_scores'][0]
            comment_time = datetime.fromisoformat(commentmetadata['created']).timestamp()

            submissiondict = eval(submissions_id2doc[submission_id])

            if subreddit not in comments_by_submission:
                comments_by_submission[subreddit] = {}
            if submission_id not in comments_by_submission[subreddit]:
                comments_by_submission[subreddit][submission_id] = {
                    'title': submissiondict['title'],
                    'text': submissiondict['body'],
                    'comments': []
                }
            
            doc = nlp(commenttext.replace("\n", " "))
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.pos_ in ['SYM', 'PUNCT', 'AUX']]
            tokens = [t for t in tokens if t.strip() != ""]
            
            comments_by_submission[subreddit][submission_id]['comments'].append({
                    'comment_id': comment_id, 
                    'comment_text': commenttext, 
                    'comment_created_utc': comment_time,
                    "comment_score": commentscore,
                    "comment_unigram": set(tokens),
                    "comment_num_sentences": len(list(doc.sents))
                })

print(f"comments_by_submission size: {len(comments_by_submission)}")

## coonvert list of paired preferences to a ranked list
def get_ranked_list(paired_prefs, key):
    item2scores = defaultdict(int)
    key2items = {}
    for item1, item2, _ in paired_prefs: #item2 is always preferred based on how this array is computed
        item2scores[item2[key]] += 1
        if item1[key] not in key2items:
            key2items[item1[key]] = item1
        if item2[key] not in key2items:
            key2items[item2[key]] = item2
    
    ranked_keyvalues = sorted([(k, v) for k, v in item2scores.items()], key=lambda x:x[1], reverse=True)
    ranked_items = [key2items[key] for key, value in ranked_keyvalues]

    return ranked_items

posts_processed = 0
data = []
foutput_pref = open(args.output_pref_file, "w")
foutput_sft = open(args.output_sft_file, "w")
total_pairs_submission = 0

#### convert to (post, commentA, commentB) for preferences
for subreddit, thread in comments_by_submission.items():
    for post_id, post in thread.items():
        history = post['text']
        posts_processed += 1
        if posts_processed % 1000 == 0:
            print("\r")
            print(f"{posts_processed/1000}K", end="", flush=True)
        unique_comment_ids = set([comment['comment_id'] for comment in post['comments']])
        assert len(unique_comment_ids) == len(post['comments']), "shame!"
        sorted_comments = sorted(post['comments'], key=lambda x: x['comment_created_utc'])
        pairs_submission = []
        special_pairs_submission = []
        for i in range(len(sorted_comments)):
            if len(sorted_comments) == 1:
                if len(sorted_comments[i]['comment_text']) > MAX_LEN: # or sorted_comments[i]['comment_score'] < MIN_SUBMISSION_SCORE: ### do this filtering later. 
                    continue
                special_pairs_submission.append([sorted_comments[i], sorted_comments[i], 1.0]) 

            for j in range(i+1, len(sorted_comments)):
                first_comment = sorted_comments[i]
                second_comment = sorted_comments[j]
                first_score, second_score = sorted_comments[i]['comment_score'], sorted_comments[j]['comment_score']
                first_len, second_len = len(sorted_comments[i]['comment_text']), len(sorted_comments[j]['comment_text'])
                
                # print(first_score, second_score)
                # print(sorted_comments[i]['comment_created_utc'], sorted_comments[j]['comment_created_utc'])
                if first_score >= second_score:
                    continue
                if second_score - first_score < MIN_SCORE_DIFF:
                    continue
                if max(first_len, second_len) > MAX_LEN:
                    continue
                
                first_unigram, second_unigram = sorted_comments[i]['comment_unigram'], sorted_comments[j]['comment_unigram']
                word_overlap = len(first_unigram.intersection(second_unigram))
                if word_overlap < MIN_WORD_OVERLAP:
                    continue

                first_num_sent, second_num_sent = sorted_comments[i]['comment_num_sentences'], sorted_comments[j]['comment_num_sentences']
                len_ratio = first_num_sent/second_num_sent if first_num_sent >= second_num_sent else second_num_sent/first_num_sent
                if len_ratio > MAX_LEN_RATIO:
                    continue

                overlap_ratio = word_overlap / len(first_unigram.union(second_unigram))
                pairs_submission.append([first_comment, second_comment, overlap_ratio])            

        #print(pairs_submission)
                
        ranked_comments = get_ranked_list(pairs_submission, "comment_id")
        special_ranked_comments = get_ranked_list(special_pairs_submission, "comment_id")
        if len(ranked_comments) > args.max_comments_per_submissions:
            ranked_comments = ranked_comments[:args.max_comments_per_submissions]
        
        for rank, comment in enumerate(ranked_comments):
            line = {
                    "rank": rank,
                    "only_comment": False,
                    "domain": subreddit,
                    "post_id": post_id,
                    "history": history,
                    "c_root_id": comment['comment_id'], 
                    "created_at_utc": comment['comment_created_utc'],
                    "score": comment['comment_score'],
                    "human_ref": comment['comment_text'],
                }
            foutput_sft.write(json.dumps(line) + "\n")
        
        for rank, comment in enumerate(special_ranked_comments):
            line = {
                    "rank": rank,
                    "only_comment": True,
                    "domain": subreddit,
                    "post_id": post_id,
                    "history": history,
                    "c_root_id": comment['comment_id'], 
                    "created_at_utc": comment['comment_created_utc'],
                    "score": comment['comment_score'],
                    "human_ref": comment['comment_text'],
                }
            foutput_sft.write(json.dumps(line) + "\n")

        if len(pairs_submission) > MAX_NUM_PAIR_PER_SUBMISSION:
            #print(f"{len(pairs_submission)} pairs from one submission over predefined max ({MAX_NUM_PAIR_PER_SUBMISSION})")
            pairs_submission = sorted(pairs_submission, key=lambda x: x[-1], reverse=True)[:MAX_NUM_PAIR_PER_SUBMISSION]
        total_pairs_submission += len(pairs_submission)
        for comment_A, comment_B, overlap_ratio in pairs_submission:
            label = int(comment_A['comment_score'] > comment_B['comment_score'])
            score_ratio = comment_A['comment_score']/comment_B['comment_score'] if comment_A['comment_score'] > comment_B['comment_score'] else comment_B['comment_score']/comment_A['comment_score']
            len_ratio = comment_A['comment_num_sentences'] / comment_B['comment_num_sentences']
            if comment_A['comment_num_sentences'] < comment_B['comment_num_sentences']:
                len_ratio = 1/len_ratio
            seconds_difference = abs(comment_A['comment_created_utc'] - comment_B['comment_created_utc'])
            # data.append(
            line = {
                    "domain": subreddit,
                    "post_id": post_id,
                    "history": history,
                    "c_root_id_A": comment_A['comment_id'], 
                    "c_root_id_B": comment_B['comment_id'],
                    "created_at_utc_A": comment_A['comment_created_utc'],
                    "created_at_utc_B": comment_B['comment_created_utc'],
                    "score_A": comment_A['comment_score'],
                    "score_B": comment_B['comment_score'],
                    "human_ref_A": comment_A['comment_text'],
                    "human_ref_B": comment_B['comment_text'],
                    "labels": label,
                    "overlap_ratio": overlap_ratio,
                    "seconds_difference": seconds_difference,
                    "score_ratio": score_ratio,
                    "len_ratio": len_ratio
                }
            foutput_pref.write(json.dumps(line) + "\n")
    
    # all_pairs += extend(pairs_submission)
# print(f"{len(pairs)} pairs from {len(comments_by_submission)} submissions")
print(f"{total_pairs_submission=}")