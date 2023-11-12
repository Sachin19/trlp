import glob
import smart_open
import json

def get_submission_ids(filepattern):
    submission_ids = []
    count = 0
    #print(filepattern)
    for filename in glob.glob(filepattern):
        # print(filename)
        with smart_open.open(filename) as f:
            for line in f:
                items = json.loads(line)
                # if f"{expname}__science_subreddit_submissions__all_pass" in items['attributes'] and\
                #     items['attributes'][f"{expname}__science_subreddit_submissions__all_pass"][0][2] > 0.5:
                #         submission_ids.append(items['id'])
                submission_ids.append(items['id'])
                if count % 1000 == 0:
                    if count > 1000000:
                        v = f"{count/1000000}m"
                    else:
                        v = f"{count/1000}k"
                    print(v, end="\r")
                count += 1
    return set(submission_ids)

#SUBMISSION_IDS = get_submission_ids("science/shp/submissions_merged*", "science_subreddits")