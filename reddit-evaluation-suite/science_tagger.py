from dolma.core.data_types import DocResult, Document, Span
from dolma import add_tagger, BaseTagger

from typing import List, Set 

from constants import SCIENCESUBREDDITS, MIN_SUBMISSION_SCORE
from science_submission_ids import get_submission_ids

@add_tagger("science_subreddit_submissions")
class ScienceSubredditSubmissionsTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        spans: List[Span] = []
        text = doc.text.lower()
        # print(dir(doc))
        linemetadata = doc.metadata
        
        keep = True
        if linemetadata['subreddit'].lower() not in SCIENCESUBREDDITS:
            keep = False
            #spans.append(Span(0, len(doc.text), type="is_about_science"))

        if keep and linemetadata['num_comments'] < 2:
            keep = False
            #spans.append(Span(0, len(doc.text), "more_than_2_comments")) 
        
        ## 0. TODO: filter posts made by a deleted user or moderator.
        ## 1. only keep posts with a score >= 10
        if keep and linemetadata['score'] < MIN_SUBMISSION_SCORE:
            keep = False
            #spans.append(Span(0, len(doc.text), "score_gt_threshold")) 

        ## 2. filter posts that were edited, deleted, hidden, pinned posts
        if keep and (linemetadata.get('edited', False) or linemetadata.get('hidden', None) or linemetadata.get('pinned', False) or linemetadata.get("removed_by", None) is not None):
            keep = False
            #spans.append(Span(0, len(doc.text), "not_edited_hidden_pinned_removed")) 

        ## 3. filter posts that are nsfw
        if keep and linemetadata.get('over_18', False):
            keep = False
            #spans.append(Span(0, len(doc.text), "sfw")) 

        ## 4. filter posts with media (photo and video)
        if keep and (linemetadata.get('is_video', False) or linemetadata.get('media', None) is not None):
            keep = False
            #spans.append(Span(0, len(doc.text), "no_media")) 

        # 5. filter posts that contain urls
        if keep and "http" in doc.text:
            keep = False
            #spans.append(Span(0, len(doc.text), "no_url"))
        
        if keep: # passes all filters
            spans.append(Span(0, len(doc.text), type="all_pass", score=1))
        else:
            spans.append(Span(0, len(doc.text), type="all_pass", score=0))


        return DocResult(doc=doc, spans=spans)

@add_tagger("science_subreddit_comments")
class ScienceSubredditCommentsTagger(BaseTagger):
    def __init__(self):
        super().__init__()
        # self.submission_ids = get_submission_ids("/net/nfs.cirrascale/allennlp/sachink/community-lm/reddit-evaluation-suite/science/submissions_merged*", "test")
        self.submission_ids = get_submission_ids("s3://ai2-llm/pretraining-data/sources/reddit/science_subreddits/submissions_merged*", "science_subreddits")
        print(len(self.submission_ids))
        ## modify

    def predict(self, doc: Document) -> DocResult:
        spans: List[Span] = []
        text = doc.text.lower()
        # print(dir(doc))
        linemetadata = doc.metadata

        keep = True
        # comment filtering
        # 0. remove comments whose posts are not in our filtered list
        if linemetadata['thread_id'] not in self.submission_ids:
           keep=False

        # ## 2. 
        # if keep and linemetadata['subreddit'].lower() not in SCIENCESUBREDDITS:
        #     keep = False

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
            spans.append(Span(0, len(doc.text), type="all_pass", score=1))
        else:
            spans.append(Span(0, len(doc.text), type="all_pass", score=0))
        
        return DocResult(doc=doc, spans=spans)