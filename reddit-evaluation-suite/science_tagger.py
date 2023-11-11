from dolma.core.data_types import DocResult, Document, Span
from dolma import add_tagger, BaseTagger

from typing import List, Set 

from constants import SCIENCESUBREDDITS, MIN_SUBMISSION_SCORE

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
            spans.append(Span(0, len(doc.text), type=doc.text, score=1))
            spans.append(Span(0, len(doc.text), type=str(doc.metadata), score=1))
        # else:
        #     spans.append(Span(0, len(doc.text), type="all_pass", score=0))


        return DocResult(doc=doc, spans=spans)
    