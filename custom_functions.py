import os.path
import numpy as np


def init_embeddings_map(fname):
    with open(os.path.join("data", "glove.6B", fname), 'r', encoding='utf-8') as glove:
        return {l[0]: np.asarray(l[1:], dtype="float32") for l in
                [line.split() for line in glove]}


def get_embed_and_pad_func(i_seq_len, u_seq_len, pad_value, embedding_map):
    def embed(row):
        sentence = row["userReviews"].strip("[']").split("', '")[:u_seq_len]
        reviews = list(map(lambda word: embedding_map.get(word)
            if word in embedding_map else pad_value, sentence))
        row["userReviews"] = reviews +\
                [pad_value] * (u_seq_len - len(reviews))
        sentence = row["movieReviews"].strip("[']").split("', '")[:i_seq_len]
        reviews = list(map(lambda word: embedding_map.get(word)
            if word in embedding_map else pad_value, sentence))
        row["movieReviews"] = reviews +\
                [pad_value] * (i_seq_len - len(reviews))
        return row
    return embed


def get_embed_aspects(aspects, embedding_map):
    embed_aspects = list(map(lambda word: embedding_map.get(word)
            if word in embedding_map else print('主题词未在Embedding_map中找到映射'), aspects))
    embed_aspects = np.array(embed_aspects)
    return embed_aspects


def get_embed_and_pad_func_for_ian(rev_text_len, asp_term_len, pad_value, embedding_map):
    def embed(row):
        sentence = row["aspect_term"].strip("[']").split("', '")[:asp_term_len]
        reviews = list(map(lambda word: embedding_map.get(word)
            if word in embedding_map else pad_value, sentence))
        row["aspect_term"] = reviews +\
                [pad_value] * (asp_term_len - len(reviews))
        sentence = row["review_text"].strip("[']").split("', '")[:rev_text_len]
        reviews = list(map(lambda word: embedding_map.get(word)
            if word in embedding_map else pad_value, sentence))
        row["review_text"] = reviews +\
                [pad_value] * (rev_text_len - len(reviews))
        return row
    return embed
