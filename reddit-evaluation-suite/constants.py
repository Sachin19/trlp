import csv

SUBSCRIBER_HIGHTHRESHOLD = 100
SUBSCRIBER_VERYHIGHTHRESHOLD = 500

fscience = csv.reader(open("sciencesubreddits.tsv"), delimiter="\t")

SCIENCESUBREDDITS = []
SCIENCESUBREDDITS_HIGHACTIVITY = []
SCIENCESUBREDDITS_AVERAGEACTIVITY = []
SCIENCESUBREDDITS_HIGHSUBSCRIBER = []
SCIENCESUBREDDITS_VERYHIGHSUBSCRIBER = []
SCIENCESUBREDDITS2DESC = {}

from collections import Counter

for line in fscience:
    SCIENCESUBREDDITS.append(line[0].strip().split("/")[-1].lower())
    SCIENCESUBREDDITS2DESC[line[0].strip().split("/")[-1].lower()] = line[-1]

    if line[3] == "Average" or line[3] == "High" or line[3] == "Very High":
        SCIENCESUBREDDITS_AVERAGEACTIVITY.append(line[0].strip().split("/")[-1].lower())

        if line[3] == "High" or line[3] == "Very High":
            SCIENCESUBREDDITS_HIGHACTIVITY.append(line[0].strip().split("/")[-1].lower())
    
    subcount = int(line[1])
    if subcount >= 100:
        SCIENCESUBREDDITS_HIGHSUBSCRIBER.append(line[0].strip().split("/")[-1].lower())

        if subcount >= 500:
            SCIENCESUBREDDITS_VERYHIGHSUBSCRIBER.append(line[0].strip().split("/")[-1].lower())
    
# counter = Counter(SCIENCESUBREDDITS)
# print([k for k, v in counter.items() if v > 1])
SCIENCESUBREDDITS = set(SCIENCESUBREDDITS)
SCIENCESUBREDDITS_HIGHACTIVITY = set(SCIENCESUBREDDITS_HIGHACTIVITY)
SCIENCESUBREDDITS_AVERAGEACTIVITY = set(SCIENCESUBREDDITS_AVERAGEACTIVITY)
SCIENCESUBREDDITS_HIGHSUBSCRIBER = set(SCIENCESUBREDDITS_HIGHSUBSCRIBER)
SCIENCESUBREDDITS_VERYHIGHSUBSCRIBER = set(SCIENCESUBREDDITS_VERYHIGHSUBSCRIBER)


# print(len(SCIENCESUBREDDITS))
# print(len(SCIENCESUBREDDITS_HIGHACTIVITY))
# print(len(SCIENCESUBREDDITS_AVERAGEACTIVITY))
# print(len(SCIENCESUBREDDITS_HIGHSUBSCRIBER))
# print(len(SCIENCESUBREDDITS_VERYHIGHSUBSCRIBER))



SCIENCESUBREDDITS_= set(
    [
        "askacademia",
        "askanthropology",
        "askbaking",
        "askcarguys",
        "askculinary",
        "askdocs",
        "askengineers",
        "askhistorians",
        "askhr",
        "askphilosophy",
        "askphysics",
        "askscience",
        "asksciencefiction",
        "asksocialscience",
        "askvet",
        "changemyview",
        "explainlikeimfive",
        "legaladvice"
    ]
)

MIN_SUBMISSION_SCORE=5

MAX_NUM_PAIR_PER_SUBMISSION=1000
MIN_WORD_OVERLAP = 1
MIN_SCORE_DIFF = 1
MAX_LEN = 10000
MAX_LEN_RATIO=5