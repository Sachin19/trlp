export AWS_ACCESS_KEY_ID="ASIASHLPW4FE4ZRHXWO5"
export AWS_SECRET_ACCESS_KEY="aSiyM/2KLDt/X4J/jIXiwElbY4tLi4uRjwezBgb0"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjELD//////////wEaCXVzLXdlc3QtMiJIMEYCIQCHIbNZIkqosFAyi6zJc2YpPu0DR3s7E3V2MdVQ2EwUGQIhAJbd7PjD30vqeI7QunSkDvLzkkA4+er7i02GobeIF/IwKvUCCOn//////////wEQBRoMMTUzMjQyNDkzMjU3Igw6TcOwV2Q63HgK3oAqyQJKRXuvaNHM37tFyOI2YotoQCvuQbFL9igo98jTd9giblAX8NTcNM6BQBERWF16ByNpSQ9BHRdZldWOKT37VJQU34TkxW6syv+7Tpbb4h1eJYrhq4BmyrrwkWsZx4LZRzqEmhxOHpswQfxVNdOSW+YxeybOLAQh+ZX97E8cGFgmUu9EiIHA3+Y1ZBrv1JDbND2Z52POldbBEG8EjH7SVRfVMzAqv/QJfFupPHh/HMxNSMyjE9nHTiExFuK+ms8+ANWPtC0u2SHCpatpbmRP7Les0SPUFlmAA26l27VM8QTkQSVpcIiJJ3DhMt4FwmVQ9lwFnMmb9mUbZvpBqL7Yz+e0tVXenVgQMKTyzvt10SB48DjOgOIcCMxRvVV7r7JUnpHOw9NpJzpno8Yq4oGynd8Y04+Uynwf4bV6+syKmsKao4y9al7YZsBBQTCsureqBjqmAUQrfT7o85O9C/iOB6ZWGvJcOB86hynlOBG27yNecUq6AiqnpU4Wv6teEAiMNl6fuWK9aLyG5L051LTWy011o5KOVrkBWEgnI1zAtqGA6h/AbNzAtKhvRY0PAFdXXSyb9WNgxw5TRhbGi1tts13ZE7Y+FBqXMbMwW6kw7qlkXfFmBWXhwvwwhaCgqmUH0aQ8NHdCzaGrz92O0FqJ8qhLRWAeIyVJfY4="

dolma tag \
    --experiment science_subreddits \
    --destination 's3://ai2-llm/pretraining-data/sources/reddit/science_subreddits/'\
    --documents \
        's3://ai2-llm/pretraining-data/sources/reddit/v0/comments_merged_*.jsonl.gz' \
    --taggers science_subreddit_comments \
    --tagger_modules /net/nfs.cirrascale/allennlp/sachink/community-lm/reddit-evaluation-suite/science_tagger.py\
    --processes 16