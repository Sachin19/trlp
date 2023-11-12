from dolma.core.data_types import DocResult, Document, Span
from dolma import add_tagger, BaseTagger

from typing import List, Set 

from constants import SCIENCESUBREDDITS_, SCIENCESUBREDDITS, SCIENCESUBREDDITS_HIGHACTIVITY, SCIENCESUBREDDITS_AVERAGEACTIVITY, SCIENCESUBREDDITS_HIGHSUBSCRIBER, SCIENCESUBREDDITS_VERYHIGHSUBSCRIBER, MIN_SUBMISSION_SCORE

from science_submission_ids import get_submission_ids

def get_spans(doc, redditset, submission_ids):
    spans: List[Span] = []
    text = doc.text.lower()
    # print(dir(doc))
    linemetadata = doc.metadata

    keep = True
    # comment filtering
    ## -1. 
    if linemetadata['subreddit'].lower() not in redditset:
        keep = False

    # 1. remove comments whose posts are not in our filtered list
    if keep and linemetadata['thread_id'] not in submission_ids:
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


@add_tagger("science_subreddit_comments")
class ScienceSubredditSubmissionsTagger(BaseTagger):
    SUBMISSION_IDS=get_submission_ids("science_subreddits_full/submissions_merged*")
    def predict(self, doc: Document) -> DocResult:
        spans = get_spans(doc, SCIENCESUBREDDITS, ScienceSubredditSubmissionsTagger.SUBMISSION_IDS)

        return DocResult(doc=doc, spans=spans)
    
@add_tagger("science_subreddit_comments_averageactivity")
class ScienceSubredditSubmissionsTaggerAA(BaseTagger):
    SUBMISSION_IDS=get_submission_ids("science_subreddits_averageactivity/submissions_merged*")
    def predict(self, doc: Document) -> DocResult:
        spans = get_spans(doc, SCIENCESUBREDDITS_AVERAGEACTIVITY, ScienceSubredditSubmissionsTaggerAA.SUBMISSION_IDS)

        return DocResult(doc=doc, spans=spans)
    
@add_tagger("science_subreddit_comments_highactivity")
class ScienceSubredditSubmissionsTaggerHA(BaseTagger):
    SUBMISSION_IDS=get_submission_ids("science_subreddits_highactivity/submissions_merged*")
    def predict(self, doc: Document) -> DocResult:
        spans = get_spans(doc, SCIENCESUBREDDITS_HIGHACTIVITY, ScienceSubredditSubmissionsTaggerHA.SUBMISSION_IDS)

        return DocResult(doc=doc, spans=spans)
    
@add_tagger("science_subreddit_comments_veryhighsubscriber")
class ScienceSubredditSubmissionsTaggerVHS(BaseTagger):
    SUBMISSION_IDS=get_submission_ids("science_subreddits_veryhighsubscriber/submissions_merged*")
    def predict(self, doc: Document) -> DocResult:
        spans = get_spans(doc, SCIENCESUBREDDITS_VERYHIGHSUBSCRIBER, ScienceSubredditSubmissionsTaggerVHS.SUBMISSION_IDS)

        return DocResult(doc=doc, spans=spans)
    
@add_tagger("science_subreddit_comments_highsubscriber")
class ScienceSubredditSubmissionsTaggerHS(BaseTagger):
    SUBMISSION_IDS=get_submission_ids("science_subreddits_highsubscriber/submissions_merged*")
    def predict(self, doc: Document) -> DocResult:
        spans = get_spans(doc, SCIENCESUBREDDITS_HIGHSUBSCRIBER, ScienceSubredditSubmissionsTaggerHS.SUBMISSION_IDS)

        return DocResult(doc=doc, spans=spans)
    
@add_tagger("shp_subreddit_comments")
class SHPSubredditSubmissionsTagger(BaseTagger):
    SUBMISSION_IDS=get_submission_ids("shp_subreddits/submissions_merged*")
    def predict(self, doc: Document) -> DocResult:
        spans = get_spans(doc, SCIENCESUBREDDITS_, SHPSubredditSubmissionsTagger.SUBMISSION_IDS)

        return DocResult(doc=doc, spans=spans)