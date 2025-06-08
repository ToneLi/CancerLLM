import json
import os
import sys
from typing import List, Callable, Tuple
import datetime

from tqdm import tqdm

from eval_util import normalize_nostem
from generation_metric import GenerationMetric, ExactMetric, BLEUMetric, ROUGEMetric
from eval_dataset_reader import PredictionsFileReader

# EFFECT_STOP_WORDS = {"and", "was", "is", "before", "afterwards", "after", "of"}

EFFECT_STOP_WORDS={""}
def get_content_from_predicted_effect(s):
    def is_stop(cand: str):
        return cand.lower() in EFFECT_STOP_WORDS

    words_prev = normalize_nostem(s).split(" ")
    words = []
    for word in words_prev:
        if not is_stop(word):
            words.append(word)
    return ' '.join(words)


def f1_emnlp2020(
        predictions: List[str],
        gold: List[str],
        generation_metric: GenerationMetric) -> Tuple[float, float, float]:
    if len(gold) == 0 and len(predictions) == 0:
        return (1.0, 1.0, 1.0)
    if len(gold) == 0 and len(predictions) > 0:
        return (0.0, 1.0, 0.0)
    if len(predictions) == 0:
        return (1.0, 0.0, 0.0)

    predictions = list(set(predictions))
    gold[0]=gold[0].replace(" .","")
    P_=[x.strip() for x in predictions[0].split(",")]
    G_=[x.strip() for x in gold[0].split(",")]

    tp = 0.0  # true positive score
    for p in predictions:
        best_gold_match = 0.0
        # print("---p",p) #
        """
        state of diet was free of addictive foods before and cleansed of addictive foods afterwards.
        
         state diet free addictive foods cleansed addictive foods
        """
        # print("p",p)
        norm_p = get_content_from_predicted_effect(p)
        # print("norm_p",norm_p)  #state food in diet eliminated
        # print("-------------------")
        for g in gold:
            norm_g = get_content_from_predicted_effect(g)
            # print("norm_g", norm_g) #availability addicting food in diet out diet

            gold_match = generation_metric.match_score(gold=norm_g, predicted=norm_p)
            # print("generation_metric",generation_metric)
            # print("----gold_match",gold_match)


            if gold_match > best_gold_match:
                best_gold_match = gold_match
        # print(f"best_gold_match :{best_gold_match}")
        tp += best_gold_match
    precision = tp / len(predictions)

    # Compute recall score based on best prediction match for each gold
    tr = 0.0
    for g in gold:
        norm_g = get_content_from_predicted_effect(g)
        best_prediction_match = 0.0
        for p in predictions:
            norm_p = get_content_from_predicted_effect(p)
            prediction_match = generation_metric.match_score(gold=norm_g, predicted=norm_p)
            if prediction_match > best_prediction_match:
                best_prediction_match = prediction_match
        tr += best_prediction_match
    recall = tr / len(gold)

    f1_denominator = precision + recall

    if f1_denominator == 0:
        return (0.0, 0.0, 0.0)

    return (precision, recall, 2 * precision * recall / (precision + recall))


class SizeMismatch(Exception):
    pass


def evaluate(predictions_reader: PredictionsFileReader,
             gold_answers_reader: PredictionsFileReader,
             diag: Callable[[str], None],
             generation_metric: GenerationMetric) -> dict:
    if len(predictions_reader.get_all_question_ids()) != len(gold_answers_reader.get_all_question_ids()):
        raise SizeMismatch(
            f"Error: Size mismatch: {predictions_reader.in_path} has {len(predictions_reader.get_all_question_ids())} predictions and \n{gold_answers_reader.in_path} has {len(gold_answers_reader.get_all_question_ids())} answers."
        )

    metric_main_p_sum = 0.0
    metric_main_r_sum = 0.0
    metric_main_f1_sum = 0.0

    all_q_ids = gold_answers_reader.get_all_question_ids()
    for q_id in tqdm(all_q_ids):
        predictions = predictions_reader.get_answers_for_id(id=q_id)

        if len(predictions) == 1 and predictions[0].lower().strip().startswith("there will be no change"):
            predictions = []
        gold_answers = gold_answers_reader.get_answers_for_id(id=q_id)

        diag("Prediction:")
        diag(predictions)
        diag("Gold:")
        diag(gold_answers)
        diag("")



        (p, r, f1) = f1_emnlp2020(
            predictions=predictions,
            gold=gold_answers,
            generation_metric=generation_metric
        )
        # print("p:",p,"r:",r,"f1:",f1)

        metric_main_p_sum += p
        metric_main_r_sum += r
        metric_main_f1_sum += f1
        diag("Instance Metrics:")
        diag(f"    Main:  P={p}, R={r}, F1={f1}")
        diag("")

    return {
        "main_P": '%.2f' % (100.0 * metric_main_p_sum / len(all_q_ids)),
        "main_R": '%.2f' % (100.0 * metric_main_r_sum / len(all_q_ids)),
        "main_F1": '%.2f' % (100.0 * metric_main_f1_sum / len(all_q_ids)),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate OpenPI predictions.')

    parser.add_argument(
        '--gold-file', '-g',
        default="/home/jiatan/G-Retriever/graph_benchmark/pharmkg_llama2_ground.jsonl",
        help='Filename with gold answers',
        required=False)

    parser.add_argument(
        '--prediction-file', '-p',
        default="/home/jiatan/G-Retriever/graph_benchmark/pharmkg_llama2_prediction.jsonl", # pharmKG_test_QA_prediction_standard.jsonl
        help='Filename with predictions',
        required=False)

    """

    ADInt_test_our_MCTS_COT_GPT4_5000_few_shot_entity_relation_sdantard.jsonl
    ADInt_test_source_MCTS_5000_zero_shot_entity_relation_sdantard.json
    ADInt_test_source_MCTS_3000_few_shot_entity_relation_sdantard
    """


    parser.add_argument(
        '--diagnostics_file', '-d',
        help='If provided, diagnostic will be printed in a file (default: %(default)s)',
        default=f"output.txt",
        required=False)

    parser.add_argument(
        '--quiet', '-q',
        help='If provided, diagnostic will not be printed or written to file.',
        action='store_true',
        required=False)

    parser.add_argument(
        '--output', '-o',
        help='Output metrics to this file in JSON format. If not specified, metrics are only printed to stdout as JSON.',
        default=None,
        required=False)

    args = parser.parse_args()
    diagnostics_file = open(args.diagnostics_file, 'w',encoding="utf-8")

    def diag(msg: str):
        if args.quiet:
            return
        diagnostics_file.write(f"{msg}")
        diagnostics_file.write("\n")

    if not args.gold_file or not os.path.exists(args.gold_file):
        print(f"WARNING: Not performing any evaluation because input gold file does not exist: {args.gold_file}")
        return

    if not args.prediction_file or not os.path.exists(args.prediction_file):
        print(f"WARNING: Not performing any evaluation because prediction file does not exist: {args.prediction_file}")
        return

    predictions = PredictionsFileReader(in_path=args.prediction_file)
    gold_answers = PredictionsFileReader(in_path=args.gold_file)
    # print("prediction",predictions)

    generation_metrics = [
        ExactMetric(),
        # BLEUMetric(),
        ROUGEMetric()
    ]
    output = open(args.output, "w", encoding="UTF-8") if args.output else sys.stdout
    #
    all_metrics = dict()
    formatted_scores = []

    for metric_num, current_metric in enumerate(generation_metrics):
        #ExactMetric
        print(f"\nEvaluating current metric ({1 + metric_num}/{len(generation_metrics)}) : {current_metric.name()} ...")

        current_metric_score = evaluate(predictions_reader=predictions,
                                        gold_answers_reader=gold_answers,
                                        diag=diag,
                                        generation_metric=current_metric
                                        )
        # print("current_metric_score",current_metric_score)
        # break
        for k, v in current_metric_score.items():
            # prepare all metrics as json entries.
            all_metrics[f"{k.replace('main_', '')}_{current_metric.name()}"] = v
        formatted_scores.append(f"{current_metric.name()}"
                                f"\t{current_metric_score['main_P']}"
                                f"\t{current_metric_score['main_R']}"
                                f"\t{current_metric_score['main_F1']}"
                                )
        # break

    if args.output:
        json.dump(all_metrics, output)

    print(f"\n\n================================\n Evaluation results \n================================")
    print(f"Predictions: {args.prediction_file}")
    print(f"Gold: {args.gold_file}")
    if not args.quiet:
        print(f"Diagnostics: {args.diagnostics_file}")
    print(f"\n\t\tprec\trecall\tf1")
    for fs in formatted_scores:
        print(fs)

    output.close()
    diagnostics_file.close()


if __name__ == '__main__':
    main()

"""

"""