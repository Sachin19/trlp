from dolma.core.data_types import DocResult, Document, Span
from dolma import add_tagger, BaseTagger

from typing import List, Set 

from constants import SCIENCESUBREDDITS, MIN_SUBMISSION_SCORE
from science_submission_ids import SUBMISSION_IDS

@add_tagger("science_subreddit_comments")
class ScienceSubredditCommentsTagger(BaseTagger):
    #def __init__(self):
    #    super().__init__()
        # self.submission_ids = get_submission_ids("/net/nfs.cirrascale/allennlp/sachink/community-lm/reddit-evaluation-suite/science/submissions_merged*", "test")
        #self.submission_ids = get_submission_ids("science/submissions_merged*", "science_subreddits")
        #print(len(self.submission_ids))
        ## modify

    def predict(self, doc: Document) -> DocResult:
        spans: List[Span] = []
        text = doc.text.lower()
        # print(dir(doc))
        linemetadata = doc.metadata

        keep = True
        # comment filtering
        ## -1. 
        if linemetadata['subreddit'].lower() not in SCIENCESUBREDDITS:
            keep = False

        # 1. remove comments whose posts are not in our filtered list
        if keep and linemetadata['thread_id'] not in SUBMISSION_IDS:
           keep=False

        ## 1. remove edited, stickied, collapsed, and self comments
        if keep and (linemetadata.get('edited', False) or linemetadata.get('stickied', False) or linemetadata.get('is_submitter', False) or linemetadata.get('collapsed', False) or linemetadata.get("score_hidden", False)):
            keep = False

        ## 2. only keep comments with a score >= 2 
        if keep and linemetadata['comment_scores'][0] < 2:
            keep = False

        ## 3. remove comments that contain urls
        user_id_spans = linemetadata['user_id_spans']
        if len(user_id_spans) > 1:
            endidx = user_id_spans[1][0]
        else:
            endidx = len(doc.text)
        if keep and "http" in doc.text[user_id_spans[0][1]+2:endidx]:
            keep = False

        if keep: # passes all filters
            spans.append(Span(0, len(doc.text), type=doc.text, score=1))
            doc.metadata['created'] = doc.created
            spans.append(Span(0, len(doc.text), type=str(doc.metadata), score=1))
            print(doc.created)
            input()
        
        return DocResult(doc=doc, spans=spans)