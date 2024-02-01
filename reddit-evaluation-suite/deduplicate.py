
import argparse
import json 
import re

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True)
parser.add_argument("--output_file", required=True)

args = parser.parse_args()

duplicates = 0
prev = ""

with open(args.input_file) as fin, open(args.output_file, "w") as fout:
    for idx, line in enumerate(fin):
        items = json.loads(line)
        post_id = items['post_id']
        # if post_id in done:
        #     continue
        post = items['history']
        # comment = items['human_ref_A']
        if "human_ref" in items:
            comment_A = items['human_ref']
            comment_B = items['human_ref']
        if "human_ref_A" in items:
            comment_A = items['human_ref_A']
            comment_B = items['human_ref_A']
        if "human_ref_B" in items:
            comment_B = items['human_ref_B']

        text = post+comment_A+comment_B
        if text == prev:
            duplicates += 0
            continue
        prev = text
        fout.write(line)
print(f"{duplicates=}")
        