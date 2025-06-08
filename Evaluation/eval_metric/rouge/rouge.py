
# -*- coding: utf-8 -*-
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
    '''
    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert(len(candidate)==1)
        assert(len(refs)>0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs/float(len(token_c)))
            rec.append(lcs/float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param hypo_for_image: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values
        :param ref_for_image: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        score = []
        for id in sorted(gts.keys()):
            hypo = res[id]
            ref  = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

        average_score = np.mean(np.array(score))
        #cconvert to percentage
        return 100*average_score, np.array(score)

    def method(self):
        return "Rouge"


class Rouge1():
    '''
    Class for computing ROUGE-1 score for a set of candidate sentences for a given dataset
    '''

    def __init__(self):
        # Similar to ROUGE-L, we can use beta for F1 score calculation
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-1 score given one candidate and references
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : reference sentences for evaluation
        :returns score: int (ROUGE-1 score for the candidate evaluated against references)
        """
        assert (len(candidate) == 1)
        assert (len(refs) > 0)

        # Initialize precision and recall lists
        prec = []
        rec = []

        # Tokenize the candidate sentence
        token_c = set(candidate[0].split(" "))

        # Calculate precision and recall for each reference
        for reference in refs:
            token_r = set(reference.split(" "))
            # Calculate the intersection (unigram overlap)
            overlap = token_c.intersection(token_r)

            # Calculate precision and recall
            prec.append(len(overlap) / float(len(token_c)))
            rec.append(len(overlap) / float(len(token_r)))

        # Take the maximum precision and recall
        prec_max = max(prec)
        rec_max = max(rec)

        # Compute F1 score using the harmonic mean of precision and recall
        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / (rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0

        return score

    def compute_score(self, gts, res):
        """
        Computes ROUGE-1 score given a set of reference and candidate sentences
        :param gts: dict : reference sentences with "image name" key and "tokenized sentences" as values
        :param res: dict : candidate/test sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-1 score for all the sentences)
        """
        scores = []
        for id in sorted(gts.keys()):
            hypo = res[id]
            ref = gts[id]

            # Calculate ROUGE-1 score for each pair of candidate and reference
            scores.append(self.calc_score(hypo, ref))

            # Sanity checks
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

        # Calculate the average score
        average_score = np.mean(np.array(scores))

        # Convert to percentage
        return 100 * average_score, np.array(scores)

    def method(self):
        return "Rouge-1"


class Rouge2():
    '''
    Class for computing ROUGE-2 score for a set of candidate sentences for a given dataset
    '''

    def __init__(self):
        # Similar to ROUGE-L, we use beta for F1 score calculation
        self.beta = 1.2

    def _get_bigrams(self, sentence):
        """
        Helper function to extract bigrams from a sentence
        :param sentence: str : input sentence
        :return: set of bigrams (2-word combinations)
        """
        tokens = sentence.split(" ")
        bigrams = set(zip(tokens, tokens[1:]))
        return bigrams

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-2 score given one candidate and references
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : reference sentences for evaluation
        :returns score: int (ROUGE-2 score for the candidate evaluated against references)
        """
        assert (len(candidate) == 1)
        assert (len(refs) > 0)

        # Initialize precision and recall lists
        prec = []
        rec = []

        # Get bigrams from the candidate sentence
        bigrams_c = self._get_bigrams(candidate[0])

        # Calculate precision and recall for each reference
        for reference in refs:
            bigrams_r = self._get_bigrams(reference)
            # Calculate the intersection (bigram overlap)
            overlap = bigrams_c.intersection(bigrams_r)

            # Calculate precision and recall
            if len(bigrams_c) > 0:
                prec.append(len(overlap) / float(len(bigrams_c)))
            else:
                prec.append(0)

            if len(bigrams_r) > 0:
                rec.append(len(overlap) / float(len(bigrams_r)))
            else:
                rec.append(0)

        # Take the maximum precision and recall
        prec_max = max(prec)
        rec_max = max(rec)

        # Compute F1 score using the harmonic mean of precision and recall
        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / (rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0

        return score

    def compute_score(self, gts, res):
        """
        Computes ROUGE-2 score given a set of reference and candidate sentences
        :param gts: dict : reference sentences with "image name" key and "tokenized sentences" as values
        :param res: dict : candidate/test sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-2 score for all the sentences)
        """
        scores = []
        for id in sorted(gts.keys()):
            hypo = res[id]
            ref = gts[id]

            # Calculate ROUGE-2 score for each pair of candidate and reference
            scores.append(self.calc_score(hypo, ref))

            # Sanity checks
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

        # Calculate the average score
        average_score = np.mean(np.array(scores))

        # Convert to percentage
        return 100 * average_score, np.array(scores)

    def method(self):
        return "Rouge-2"
