import glob
import smart_open
import json
import boto3

def get_submission_ids(filepattern)#, experiment="gender_subreddits", submissions_tagger_name="gender_subreddit_submissions"):
    print(filepattern)
    submission_ids = []
    count = 0
    #print(filepattern)

    protocol, submission_pattern = filepattern.split("://")
    items = submission_pattern.split("/")
    bucket = items[0]
    prefix = "/".join(items[1:]).split("*")[0]
    # print(protocol, bucket, prefix)

    client=boto3.client(protocol)
    submission_objs = client.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']
    # submissions_files = glob.glob(args.submissions_file_pattern)
    submission_files = [f"{protocol}://{bucket}/{obj['Key']}" for obj in submission_objs]
    print(len(submission_files))
    subreddits = set()
    # submission_prefix_len = len(f"{experiment}__{submissions_tagger_name}__")
    for filename in submission_files:
        # print(filename)
        with smart_open.open(filename) as f:
            for line in f:
                items = json.loads(line)
                # if f"{expname}__science_subreddit_submissions__all_pass" in items['attributes'] and\
                #     items['attributes'][f"{expname}__science_subreddit_submissions__all_pass"][0][2] > 0.5:
                #         submission_ids.append(items['id'])
                submission_ids.append(items['id'])
                # subreddits.add(items['subreddit'])
                # submissiondict = json.loads(submissiondoc)
                # attributes = list(items['attributes'].keys())
                # #print(attributes)
                # submissions_id2doc[submissiondict['id']] = attributes[1][submission_prefix_len:] #metadata
                # subdoc = eval(attributes[1][submission_prefix_len:])
                # all_submission_attributes.update(list(subdoc.keys()))
                # subreddits.add(subdoc['subreddit'])
                # total_submissions += 1
            # if total_submissions % 1000 == 0:
            #     print(f"{total_submissions/1000}K", end="\r", flush=True)
                if count % 1000 == 0:
                    if count > 1000000:
                        v = f"{count/1000000}m"
                    else:
                        v = f"{count/1000}k"
                    print(v, end="\r")
                count += 1
    print(subreddits)
    return set(submission_ids)

#SUBMISSION_IDS = get_submission_ids("science/shp/submissions_merged*", "science_subreddits")