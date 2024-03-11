import csv
from collections import Counter

SUBSCRIBER_HIGHTHRESHOLD = 100
SUBSCRIBER_VERYHIGHTHRESHOLD = 500

factor2filename = {
    "gender": "redditlists/gendersubreddits.tsv",
    "history": "redditlists/historysubreddits.tsv",
    "finance": "redditlists/financesubreddits.tsv",
    "science": "redditlists/sciencesubreddits.tsv",
    "politics": "redditlists/politicssubreddits.tsv"
}


def get_subreddits(factor):
    filename = factor2filename[factor]
    fsubreddits = csv.reader(open(filename), delimiter="\t")
    # print(fsubreddits)
    SUBREDDITS = []
    # SUBREDDITS_HIGHACTIVITY = []
    # SUBREDDITS_AVERAGEACTIVITY = []
    # SUBREDDITS_HIGHSUBSCRIBER = []
    # SUBREDDITS_VERYHIGHSUBSCRIBER = []
    # SUBREDDITS2DESC = {}

    for line in fsubreddits:
        if len(line) > 0:
            SUBREDDITS.append(line[0].strip().split("/")[-1].lower())
        # SUBREDDITS2DESC[line[0][3:]] = line[-1]

        # if line[3] == "Average" or line[3] == "High" or line[3] == "Very High":
        #     SUBREDDITS_AVERAGEACTIVITY.append(line[0][3:])

        #     if line[3] == "High" or line[3] == "Very High":
        #         SUBREDDITS_HIGHACTIVITY.append(line[0][3:])
        
        # subcount = int(line[1])
        # if subcount >= 100:
        #     SUBREDDITS_HIGHSUBSCRIBER.append(line[0][3:])

        #     if subcount >= 500:
        #         SUBREDDITS_VERYHIGHSUBSCRIBER.append(line[0][3:])
        
    # counter = Counter(SUBREDDITS)
    # print([k for k, v in counter.items() if v > 1])
    SUBREDDITS = set(SUBREDDITS)
    # SUBREDDITS_HIGHACTIVITY = set(SUBREDDITS_HIGHACTIVITY)
    # SUBREDDITS_AVERAGEACTIVITY = set(SUBREDDITS_AVERAGEACTIVITY)
    # SUBREDDITS_HIGHSUBSCRIBER = set(SUBREDDITS_HIGHSUBSCRIBER)
    # SUBREDDITS_VERYHIGHSUBSCRIBER = set(SUBREDDITS_VERYHIGHSUBSCRIBER)


    # print(len(SUBREDDITS))
    # print(len(SUBREDDITS_HIGHACTIVITY))
    # print(len(SUBREDDITS_AVERAGEACTIVITY))
    # print(len(SUBREDDITS_HIGHSUBSCRIBER))
    # print(len(SUBREDDITS_VERYHIGHSUBSCRIBER))
    return SUBREDDITS 

MIN_SUBMISSION_SCORE=5

MAX_NUM_PAIR_PER_SUBMISSION=1000
MIN_WORD_OVERLAP = 1
MIN_SCORE_DIFF = 1
MAX_LEN = 10000
MAX_LEN_RATIO=5