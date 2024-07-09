import logging

import Levenshtein

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Levenshtein similarity functions
def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    if o_q_i.endswith('.'):
        o_q_i = o_q_i[:-1]
    if a_ij.endswith('.'):
        a_ij = a_ij[:-1]
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(
        predicted_answers
    ), "Length of ground_truth and predicted_answers must match."

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = ground_truth[i]
        o_q_i = predicted_answers[i]
        if o_q_i == "":
            logger.warning("Skipped an empty prediction.")
            max_score = 0
        else:
            max_score = max(similarity_score(a_ij.lower(), o_q_i.lower()) for a_ij in a_i)
        total_score += max_score

    return total_score / N
