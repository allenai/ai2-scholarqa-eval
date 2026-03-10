#!/usr/bin/env python3

import json
import numpy as np
import argparse
import scipy.stats as stats
import pandas as pd
import math
import sys
import os

def load_and_concatenate_jsons(human_agreed_only=False):
    if human_agreed_only:
        files = [('deep_expert', "annotation/pairwise_authored.json"),
                 ('near_expert', "annotation/pairwise_chosen.json"),
                 ('dev_assigned', "annotation/pairwise_dev_assigned_agreement_only.json"),
                 ('test_assigned', "annotation/pairwise_test_assigned_agreement_only.json"),
        ]
    else:
        files = [('deep_expert', "annotation/pairwise_authored.json"),
                 ('near_expert', "annotation/pairwise_chosen.json"),
                 ('dev_assigned_1', "annotation/pairwise_dev_assigned_ann1.json"),
                 ('dev_assigned_2', "annotation/pairwise_dev_assigned_ann2.json"),
                 ('test_assigned_1', "annotation/pairwise_test_assigned_ann1.json"),
                 ('test_assigned_2', "annotation/pairwise_test_assigned_ann2.json"),
        ]

    combined_data = []
    models = []
    for source, file in files:
        try:
            with open(os.path.join(os.path.dirname(__file__),file), 'r') as f:
                data = json.load(f)

            for entry in data:
                models.extend(entry['models'])
                entry['source'] = source
                combined_data.append(entry)

            print(f"Loaded {len(data)} entries from {file}")
        except FileNotFoundError:
            print(f"File {file} not found.")
    print(f"Total entries: {len(combined_data)}\n")

    return combined_data, set(models)


def filter_by_source(data, sources):
    return [entry for entry in data if entry.get('source') in sources]


def filter_by_source_and_reference_question(data, source, ref):
    ref_questions = [entry['question'] for entry in data if entry.get('source') == ref]
    return [entry for entry in data if entry.get('source') == source and entry.get('question') in ref_questions]


def calculate_exact_tie_proportion(data, metric_name):
    """Calculate proportion of exact ties for a given metric."""
    ties = 0
    total = 0

    for entry in data:
        if metric_name in entry:
            scores = entry[metric_name]
            if len(scores) == 2:
                total += 1
                if scores[0] == scores[1]:
                    ties += 1

    proportion = ties / total if total > 0 else 0
    return proportion, ties, total


def calculate_threshold_tie_proportion(data, metric_name, threshold):
    """Calculate proportion of ties for a given metric using a threshold."""
    ties = 0
    total = 0

    for entry in data:
        if metric_name in entry:
            scores = entry[metric_name]
            if None in scores:
                continue
            if len(scores) == 2:
                total += 1
                if abs(scores[0] - scores[1]) <= threshold:
                    ties += 1

    proportion = ties / total if total > 0 else 0
    return proportion


def find_optimal_threshold(data, human_metric, model_metric, thresholds):
    """Find the threshold that produces model ties closest to human ties."""
    # Get human baseline
    human_tie_prop, human_ties, human_total = calculate_exact_tie_proportion(data, human_metric)

    # Test all thresholds
    results = []
    for threshold in thresholds:
        model_tie_prop = calculate_threshold_tie_proportion(data, model_metric, threshold)
        diff = abs(model_tie_prop - human_tie_prop)
        results.append({
            'threshold': threshold,
            'model_tie_proportion': model_tie_prop,
            'difference': diff
        })

    # Find optimal threshold
    optimal = min(results, key=lambda x: x['difference'])

    return {
        'human_metric': human_metric,
        'model_metric': model_metric,
        'human_tie_proportion': human_tie_prop,
        'human_ties': human_ties,
        'human_total': human_total,
        'optimal_threshold': optimal['threshold'],
        'model_tie_proportion': optimal['model_tie_proportion'],
        'difference': optimal['difference'],
        'all_thresholds': results
    }


def get_decision(score1, score2, threshold=None):
    """
    Returns: 1 if score1 > score2, -1 if score1 < score2, 0 otherwise
    """
    if threshold is None:
        # Human evaluation - only exact matches are ties
        if score1 == score2:
            return 0
        elif score1 > score2:
            return 1
        else:
            return -1
    else:
        # Model evaluation - use threshold
        if abs(score1 - score2) <= threshold:
            return 0
        elif score1 > score2:
            return 1
        else:
            return -1

def calculate_agreement(data, human_metric, model_metric, threshold, drop_elicit=False, tie_strategy="threshold", subselect_models=None):
    model_human_agreements = 0
    total = 0
    human = []
    model = []

    # for tau
    concordant = 0
    discordant = 0
    tied_human = 0
    tied_model = 0

    if tie_strategy == 'partial':
        threshold = None

    for entry in data:
        if drop_elicit and 'elicit' in entry['models']:
            continue

        if subselect_models is not None and (entry['models'][0] not in subselect_models or entry['models'][
                1] not in subselect_models):
            continue

        if human_metric not in entry or model_metric not in entry:
            continue
        human_scores = entry[human_metric]
        model_scores = entry[model_metric]

        if None in model_scores:
            continue


        # Get decisions
        human_decision = get_decision(human_scores[0], human_scores[1])
        if human_decision == 0 and tie_strategy=='exclude':
            continue

        if threshold is None:
            model_decision = get_decision(model_scores[0], model_scores[1], 0)
        else:
            model_decision = get_decision(model_scores[0], model_scores[1], threshold)

        # track agreements
        human.append(human_decision)
        model.append(model_decision)


        total += 1

        if human_decision == model_decision:
            model_human_agreements += 1
            if human_decision in [1, -1]:
                concordant += 1
            else:
                tied_human += 1
                tied_model += 1
        elif human_decision == 0:
            if tie_strategy=='partial':
                model_human_agreements += .5
            tied_human += 1
        elif model_decision == 0:
            if tie_strategy=='partial':
                model_human_agreements += .5
            tied_model += 1
        else:
            discordant += 1

    try:
        tau = (concordant - discordant) / math.sqrt((total-tied_human)*(total-tied_model))
    except ZeroDivisionError:
        tau = 0.0

    agreement_rate = model_human_agreements / total if total > 0 else 0
    return agreement_rate, model_human_agreements, tau, total


def calculate_agreement_two_annot(data1, data2, human_metric, model_metric, threshold, drop_elicit=False, tie_strategy="threshold", subselect_models=None):
    a1 = calculate_agreement(data1, human_metric, model_metric, threshold, drop_elicit=drop_elicit, tie_strategy=tie_strategy, subselect_models=subselect_models)
    a2 = calculate_agreement(data2, human_metric, model_metric, threshold, drop_elicit=drop_elicit, tie_strategy=tie_strategy, subselect_models=subselect_models)

    return calc_mean(a1[0],a2[0]), calc_mean(a1[1],a2[1]), calc_mean(a1[2],a2[2]), calc_mean(a1[3],a2[3])


def calc_mean(a,b):
    return (a+b) / 2


def run_tie_calibration(data, metric_pairs, all_results):
    # Default behavior - threshold analysis
    print("=" * 70)
    print("HUMAN TIE PROPORTIONS (BASELINE)")
    print("=" * 70)
    for human_metric, _ in metric_pairs:
        prop, ties, total = calculate_exact_tie_proportion(data, human_metric)
        print(f"{human_metric:30s}: {prop:.4f} ({ties}/{total} ties)")

    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLDS FOR MODEL METRICS")
    print("=" * 70)

    for result in all_results:
        print(f"\n{result['human_metric']} -> {result['model_metric']}")
        print(f"  Human tie proportion: {result['human_tie_proportion']:.4f}")
        print(f"  Optimal threshold: {result['optimal_threshold']:.2f}")
        print(f"  Model tie proportion at optimal: {result['model_tie_proportion']:.4f}")
        print(f"  Difference: {result['difference']:.4f}")

    # Print detailed threshold analysis
    print("\n" + "=" * 70)
    print("DETAILED THRESHOLD ANALYSIS")
    print("=" * 70)

    for result in all_results:
        print(f"\n{result['model_metric']}")
        print(f"  (Target: {result['human_tie_proportion']:.4f})")
        print(f"  {'Threshold':<12} {'Model Ties':<15} {'Difference':<12}")
        print(f"  {'-' * 40}")

        for thresh_result in result['all_thresholds']:
            marker = " <-- OPTIMAL" if thresh_result['threshold'] == result['optimal_threshold'] else ""
            print(f"  {thresh_result['threshold']:<12.2f} "
                  f"{thresh_result['model_tie_proportion']:<15.4f} "
                  f"{thresh_result['difference']:<12.4f}{marker}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Metric Pair':<50} {'Optimal Threshold':<20}")
    print("-" * 70)
    for result in all_results:
        pair_name = f"{result['human_metric']} -> {result['model_metric']}"
        print(f"{pair_name:<50} {result['optimal_threshold']:<20.2f}")


def run_metric(setting, comparison_metric_pairs, threshold_per_metric, use_threshold=False, tie_strategy="threshold", data = None, data_1 = None, data_2 = None, drop_elicit=False, subselect_models=None):
    save_for_print = []
    for human_metric, model_metric in comparison_metric_pairs:
        threshold = threshold_per_metric[model_metric]

        if data is None:
            agreement_rate, agreements, tau, total = calculate_agreement_two_annot(
                data_1, data_2, human_metric, model_metric, threshold=threshold if use_threshold else None, drop_elicit=drop_elicit, tie_strategy=tie_strategy, subselect_models=subselect_models
                )
        else:
            agreement_rate, agreements, tau, total = calculate_agreement(
                data, human_metric, model_metric, threshold=threshold if use_threshold else None, drop_elicit=drop_elicit, tie_strategy=tie_strategy, subselect_models=subselect_models
                )

        save_for_print.append({
            'human_metric': human_metric,
            'model_metric': model_metric,
            'size': total,
            'n': agreements,
            'agreement': agreement_rate,
            'tau': tau,
            'threshold_used': threshold,
            })
    df = pd.DataFrame(save_for_print)
    print(df.to_string())
    df['setting'] = setting

    return(df)


def calculate_using_thresholds(data_for_threshold, metric_pairs):
    # Define threshold range
    thresholds = np.arange(0.01, 0.16, 0.01)

    all_results_thresholds = []
    threshold_per_metric = {}

    for human_metric, model_metric in metric_pairs:
        result = find_optimal_threshold(data_for_threshold, human_metric, model_metric, thresholds)
        all_results_thresholds.append(result)
        threshold_per_metric[result['model_metric']] = result['optimal_threshold']

    return all_results_thresholds, threshold_per_metric


def average_thresholds(thresholds_per_metric):
    sums = {key: 0.0 for key in thresholds_per_metric[0]}

    for d in thresholds_per_metric:
        for key, value in d.items():
            sums[key] += value

    averages = {key: total / len(thresholds_per_metric) for key, total in sums.items()}
    return averages


def system_rankings(data, drop_elicit = False):
    wins = {}
    all_models = set([model for item in data for model in item['models']])

    total = 0
    for item in data:
        if drop_elicit and 'elicit' in item['models']:
            continue

        total += 1
        for metric in ['model_overall', 'human_overall', 'preference_ranking_norm']:
            if metric not in item:
                continue
            if metric not in wins:
                wins[metric] = {}
                for model in all_models:
                    if model not in wins[metric]:
                        wins[metric][model] = []
            a,b = item[metric]
            model_a = item['models'][0]
            model_b = item['models'][1]
            if a > b:
                wins[metric][model_a].append(1)
                wins[metric][model_b].append(0)
            elif a < b:
                wins[metric][model_a].append(0)
                wins[metric][model_b].append(1)
            else:
                wins[metric][model_a].append(.5)
                wins[metric][model_b].append(.5)

    tally = []
    for metric,model_scores in wins.items():
        for model_name, scores in model_scores.items():
            if drop_elicit and model_name == 'elicit':
                continue
            winses = sum(1 for s in scores if s == 1)
            losses = sum(1 for s in scores if s == 0)
            ties = sum(1 for s in scores if s == 0.5)
            win_rate = sum(scores) / len(scores)

            tally.append({
                'metric': metric,
                'model': model_name,
                'wins': winses,
                'losses': losses,
                'ties': ties,
                'win_rate': win_rate
                })

    df = pd.DataFrame(tally)
    m = df[df['metric'] == 'model_overall'].win_rate.to_list()
    h = df[df['metric'] == 'human_overall'].win_rate.to_list()
    p = df[df['metric'] == 'preference_ranking_norm'].win_rate.to_list()
    models = df['model'].to_list()

    system_corr_preference = stats.kendalltau(m, p)[0]
    # print(f'Model vs Human Preference: {stats.kendalltau(m, p)[0]:.3}')
    # if len(h) > 0:
    #     print(f'Model vs Human Overall: {stats.kendalltau(m,h)[0]:.3}')

    dfout = pd.DataFrame([models,p,m]).T.dropna().sort_values(1, ascending=False)
    dfout.columns = ['generators','human','model']
    return dfout, system_corr_preference

def calculate_iaa(annotator_data, drop_elicit = False, subselect_models=None):
    annotator_choice = {}
    for i, a_data in enumerate(annotator_data):
        for entry in a_data:
            if drop_elicit and 'elicit' in entry['models']:
                continue

            if subselect_models is not None and (entry['models'][0] not in subselect_models or entry['models'][
                1] not in subselect_models):
                continue

            if entry['question'] not in annotator_choice:
                annotator_choice[entry['question']] = [None, None]

            annotator_choice[entry['question']][i] = get_decision(entry['preference_ranking_norm'][0],
                                                                          entry['preference_ranking_norm'][1])

    tally = []
    tally_half_credit = []
    for k, v in annotator_choice.items():
        if v[0] == v[1]:
            tally.append(1)
            tally_half_credit.append(1)
        else:
            if v[0] == 0 or v[1] == 0:
                tally_half_credit.append(0.5)
            tally.append(0)
    return tally, tally_half_credit

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze tie proportions and agreement rates between human and model evaluations.'
    )
    parser.add_argument(
        '--tie_strategy',
        type=str,
        choices=['threshold', 'exclude', 'partial'],
        default='threshold',
        help='What to do with ties?'
        )
    parser.add_argument(
        '--dont_use_thresholds',
        action='store_true',
        help='turn off thresholding')

    parser.add_argument(
        '--subselect_models',
        type=str,
        default='',
        help='comma delimited list of models to run. Default: sqa,openai-dr,perplexity'
    )
    parser.add_argument(
        '--human_agreed',
        action='store_true',
        help='Only calculate for human agreed instances'
        )
    parser.add_argument(
        '--drop_elicit',
        action='store_true',
        help='drop elicit from system ranking'
        )
    parser.add_argument(
        '--source',
        type=str,
        choices=['deep_expert', 'near_expert', 'assigned'],
        default='assigned',
        help='Filter by data source: deep_expert, near_expert, or assigned (test+dev)'
    )
    args = parser.parse_args()


    # Define metric pairs
    metric_pairs = [
        ('human_overall', 'model_overall'),
        ('human_answer_precision', 'model_answer_precision'),
        ('human_answer_recall', 'model_answer_recall'),
        ('human_citation_precision', 'model_citation_precision'),
        ('human_citation_recall', 'model_citation_recall'),
        ]

    overall_pairwise_preference = [
        ('preference_ranking_norm', 'model_overall'),
        ('preference_ranking_norm', 'model_answer_precision'),
        ('preference_ranking_norm', 'model_answer_recall'),
        ('preference_ranking_norm', 'model_citation_precision'),
        ('preference_ranking_norm', 'model_citation_recall'),
    ]
    metric_wise = [
        ('human_overall', 'model_overall'),
        ('human_answer_precision', 'model_answer_precision'),
        ('human_answer_recall', 'model_answer_recall'),
        ('human_citation_precision', 'model_citation_precision'),
        ('human_citation_recall', 'model_citation_recall'),
        ]


    print("="*70)
    print("LOADING DATA")
    print("="*70)

    data_unfiltered,all_model_list = load_and_concatenate_jsons(human_agreed_only=args.human_agreed)

    if len(data_unfiltered) == 0:
        print("No data found.")
        sys.exit(1)

    subselect_models = args.subselect_models.split(',') if args.subselect_models != '' else None
    if subselect_models:
        for m in subselect_models:
            if m not in all_model_list:
                print('Subselected model "' + m + '" not found.')
                exit(1)


    data_annotator_1 = None
    data_annotator_2 = None

    # Filter by source
    if args.source == "assigned":
        data_annotator_1 = filter_by_source(data_unfiltered, ['dev_assigned_1', 'test_assigned_1'])
        data_annotator_2 = filter_by_source(data_unfiltered, ['dev_assigned_2', 'test_assigned_2'])

        all_results_thresholds_1, threshold_per_metric_1 = calculate_using_thresholds(data_annotator_1, metric_pairs)
        all_results_thresholds_2, threshold_per_metric_2 = calculate_using_thresholds(data_annotator_2, metric_pairs)

        threshold_per_metric = average_thresholds([threshold_per_metric_1, threshold_per_metric_2])

        if args.human_agreed:
            data = filter_by_source(data_unfiltered, ['dev_assigned', 'test_assigned'])
        else:
            data = None

    else:
        data_for_threshold = filter_by_source(data_unfiltered, ['deep_expert', 'near_expert'])
        all_results_thresholds, threshold_per_metric = calculate_using_thresholds(data_for_threshold, metric_pairs)

        data = filter_by_source(data_unfiltered, [args.source])


    if data is None: # if doubly annotated
        print("=" * 70)
        print("IAA")
        print("=" * 70)
        iaa,iaa_hc = calculate_iaa([data_annotator_1, data_annotator_2], drop_elicit=args.drop_elicit, subselect_models=subselect_models)

        print(f'Strict: {sum(iaa)/len(iaa)}')
        print(f'Half Credit: {sum(iaa_hc)/len(iaa_hc)}')


    print("="*70)
    print(f"Pairwise Agreements: {args.source} {'(Dropping ELICIT)' if args.drop_elicit else ''}")
    print("="*70)

    if args.dont_use_thresholds:
        use_threshold = False
    else:
        use_threshold = True

    # Calculate the metrics
    all_save = []
    df = run_metric('optimal_threshold',
                    overall_pairwise_preference,
                    threshold_per_metric,
                    use_threshold=use_threshold,
                    tie_strategy=args.tie_strategy,
                    data=data,
                    data_1=data_annotator_1,
                    data_2=data_annotator_2,
                    drop_elicit=args.drop_elicit,
                    subselect_models=subselect_models,)
    all_save.append(df)

    print()
    print("="*70)
    print(f"Metric-wise: {args.source} {'(Dropping ELICIT)' if args.drop_elicit else ''}")
    print("="*70)

    df2 = run_metric('optimal_threshold',
                    metric_wise,
                    threshold_per_metric,
                    use_threshold=use_threshold,
                    tie_strategy=args.tie_strategy,
                    data=data,
                    data_1=data_annotator_1,
                    data_2=data_annotator_2,
                    drop_elicit=args.drop_elicit,
                    subselect_models=subselect_models,)
    all_save.append(df2)

    # print()
    # print(pd.concat(all_save, ignore_index=True).to_csv())

    # Load data
    print()
    print("="*70)
    print(f"System rankings (Human Preference vs. Model Overall) {'(Dropping ELICIT)' if args.drop_elicit else ''}")
    print("="*70)


    if data is not None:
        df_system, corr = system_rankings(data, drop_elicit = args.drop_elicit)
        print(f'Corr (kendal tau): {corr:0.2f}')
    else:
        df_system1, corr1 = system_rankings(data_annotator_1, drop_elicit = args.drop_elicit)
        df_system2, corr2 = system_rankings(data_annotator_2, drop_elicit = args.drop_elicit)
        print(f'Annot 1: {corr1:0.2f}')
        print(f'Annot 2: {corr2:0.2f}')
        print(f'Annotator average corr (kendall tau): {(corr1+corr2)*0.5:0.2f}')

        df_system = pd.merge(df_system1, df_system2[['generators','human']], on="generators")
        df_system['human'] = (df_system['human_x'] + df_system['human_y']) / 2

    print()
    print(df_system[['generators','human','model']])


if __name__ == "__main__":
    main()
