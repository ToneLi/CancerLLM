

import re
import string

from eval_metric.bleu.bleu import Bleu
from eval_metric.rouge.rouge import Rouge,Rouge1,Rouge2
from eval_util import normalize_and_stem


class GenerationMetric:
    def __init__(self):
        self.metric = None

    def match_score(self, gold: str, predicted: str):
        if not gold and not predicted:
            # both empty then return 1.0
            return 1.0

        if not gold: # gold empty then return 0.0
            return 0.0

        if not predicted: # predicted empty then return 0.0
            return 0.0

        return self.compute_score(gold=gold, predicted=predicted)

    def compute_score(self, gold: str, predicted: str):
        pass

    def name(self):
        pass


class ExactMetric(GenerationMetric):
    def __init__(self):
        self.metric = None

    def compute_score(self, gold: str, predicted: str):
        # print(gold)
        # print(predicted)
        # print("-----")
        return 1.0 if normalize_and_stem(gold) == normalize_and_stem(predicted) else 0.0

    def name(self):
        return "ExactMetric"

import openai
openai.api_key = ''

SYSTEM_PROMPT="You are an assistant who evaluates medical term similarity"
USER_PROMPT_1 = "Are you clear about your role?"
ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with evaluation task"

def openai_chat_completion_response(final_prompt):
    response = openai.chat.completions.create(
        timeout=600,
        max_tokens=350,
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_1},
            {"role": "assistant", "content": ASSISTANT_PROMPT_1},
            {"role": "user", "content": final_prompt}
        ]
    )

    return response.choices[0].message.content.strip(" \n")


def gpt4evaluation(term1, term2):
    prompt = f"Are the medical terms '{term1}' and '{term2}' similar? Please respond with 'Yes' or 'No'."

    predictions = openai_chat_completion_response(prompt)
    return predictions


class GPT4Metric(GenerationMetric):
    def __init__(self):
        self.metric = None

    def compute_score(self, gold: str, predicted: str):
        # print(gold)
        # print(predicted)
        # print("-----")
        if "yes" in gpt4evaluation(normalize_and_stem(gold),normalize_and_stem(predicted) ).lower():
            return 1.0
        else:
            return 0.0

        # return 1.0 if normalize_and_stem(gold) == normalize_and_stem(predicted) else 0.0

    def name(self):
        return "GPT4Metric"


class NormStrMetric(GenerationMetric):
    def __init__(self):
        self.metric = None

    @staticmethod
    def normalizer(s: str) -> str:
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_score(self, gold: str, predicted: str):
        return 1.0 if normalize_and_stem(gold) == normalize_and_stem(predicted) else 0.0

    def name(self):
        return "NormStrMetric"


class BLEUMetric(GenerationMetric):
    def __init__(self, n: int=4):
        self.metric = Bleu(n)

    def compute_score(self, gold: str, predicted: str):
        # Reference: gold , Hypothesis: predicted
        # print("--------------ddd")
        # print("gold",gold)
        # print("predicted",predicted)
        score, score_info = self.metric.compute_score(
            gts={0: [normalize_and_stem(gold)]},
            res={0: [normalize_and_stem(predicted)]}
        )
        # print("BLEUMetricscore",score)
        #return scores[3] # scores[3] represents BLEU4, in this experiment,we only use bleu4
        # return average of Bleu_1, Bleu_2, Bleu_3, Bleu_4
        # return sum(score)/len(score)/100.0
        # print("=================")
        # print("ffff score",score)
        return score[1]/100.0

    def name(self):
        return "BLEUMetric"


class ROUGEMetric(GenerationMetric):
    def __init__(self):
        self.metric = Rouge1()

    def compute_score(self, gold: str, predicted: str):
        # Reference: gold , Hypothesis: predicted
        score, score_info = self.metric.compute_score(
            gts={0: [normalize_and_stem(gold)]},
            res={0: [normalize_and_stem(predicted)]}
        )

        return score/100.0

    def name(self):
        return "ROUGE1_1_Metric"


