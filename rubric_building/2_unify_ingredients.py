import yaml
import pandas as pd
import json
import random
import os
import tqdm
import time

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


from utils_rubric_building import parse_json, make_batch_request


SYSTEM_PROMPT = """I will give you a user query and a list of ingredients. The ingredients are written requirements for writing a good answer. Note that ingredients the writer thought are more critical to answering the query are prefixed with "The answer SHOULD". Useful but not critical information is marked as "The answer MIGHT".
Do the following:
1. Identify the key concepts, ideas, and named entities that should be covered for this question
2. Carefully consider the query and the ingredients given to you. At this stage, ONLY look at the ingredient description (do not consider the examples) to identify a minimal set of non-overlapping key requirements that either are high-quality ingredients OR are consistently being covered in the ingredient list. Take into consideration concepts identified in 1, especially when deciding if the key requirement should be a “SHOULD” or “MIGHT” requirement.
3. Next, step through each of the given ingredients, and decide which set requirements it should be associated with, and distribute the examples (see Notes 1 and 2).
4. Prune the examples: Remove exact or near duplicates. Remove examples that you judge are not directly relevant to the key requirement.
5. Finally, list ingredients that were left out and why.

Note1: You are allowed and encouraged to place multiple ingredients into a single key requirement. This would be fitting in the case of duplicate or near duplicate ingredients like "discuss physical commonsense datasets like PIQA" vs. "include a discussion of PIQA or other physical commonsense datasets". This type of grouping can also happen if you have a more general key requirement that can handle multiple ingredients, for example, for a key requirement "discuss success of AI in disease detection" might encompass ingredients like "mention AI success in diabetic retinopathy prediction" and "point out that machine learning methods have been successfully used on ECG data to identify early signs of atrial fibrillation".
Note2: You are allowed to split ingredients into multiple key requirements. For example, if an ingredient reads "The answer might explain why the engagement dropped, focusing on common mistakes in interface design.", you may end up placing it under both the requirement "The answer might explain the drop in engagement" and the requirement "The answer might discuss common mistakes in interface design", distributing its examples to the appropriate requirement.

Rules:
* Always keep your focus on the query. All key requirements must be relevant for the query.
* NEVER include an ingredient in a requirement on the basis of the examples alone. ALWAYS make sure that the ingredient description is prioritized.
* Use your best judgement for deciding whether a key requirement should be a “SHOULD” or “MIGHT” requirement ALWAYS based on the question and the key concepts and ideas you identified early on.
* Each requirement should ideally address a different component of the query. If the query requests “Effect of phonemic perceptions is evident in language acquisition, speech comprehension, and second language learning”, a single requirement shouldn’t try to address all three “language acquisition”, “speech comprehension”, and “second language learning”. Ideally these should be separated out into multiple requirements.
* Remember, the key requirements should not be overlapping. For example: Note that ingredient R1-“The answer should introduce transformer architecture components, including attention mechanisms and their role in sequence modeling” partially overlaps with R2-“The answer should discuss the role of attention mechanisms in sequence modeling”. This should be avoided, when possible: R1 could instead be “The answer should introduce transformer architecture components” since the rest is covered by R2.
* Each key requirement should be self-contained and understandable without needing to know about other requirements (e.g. pronouns like "these" in "should further describe these approaches" that refer to the previous requirements should be avoided and be replaced with mentions).
* Although “should” ingredients are more important, the “might” ingredients are also valuable to Include those that you think they would (best) help answering the user's query.
* There should never be a key requirement that has no ingredient associated.
* It’s okay to have leftover ingredients. Ingredients that you think are not very relevant, too vague, or peripherally relevant can be left out even if they carry the "should" phrasing.
* Background or causally related information unless the query asks explicitly for them, should be considered "MIGHT" requirements.
* DO NOT include key requirements that are centrally about paper citations. For example, do not include requirements like "List recent papers..." or "Cite the most impactful papers..." or "Identify and discuss important papers...".

Repeat (THINK) after me!
* I will be choosy about "SHOULD" requirements. "MIGHT" requirements, I can use liberally.
* I will base "SHOULD" and "MIGHT" based on key concepts I judge as being central to answering the query.
* I will always write requirements that are relevant to the query.

Return a json:
{
"key_requirements": [
{
"key_requirement": description designed after the ingredients you group together,
"ingredients": [the ingredient id list of those ingredients you grouped.],
"examples": [concatenated relevant examples from ingredients in this requirement { "detail": examples/details if relevant, "citation": citation if available; null if not available }, ...]
},
...
]
"left_out_ingredients": [
{"ingredient": id of the ingredient that got left out, "reason": brief reason why it was left out.}, ...
]
}
"""

default_log_dir = 'logs_unification'

def main(config: dict):
    log_dir = os.path.join(os.path.dirname(__file__),config['log_dir'] if config['log_dir'] != '' else default_log_dir)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M")
    with open(config['all_ingredients_file']) as f:
        df = pd.read_json(f, lines=True)
    print('Models:', config['unifier_models'])
    all_solvers = set(df.solver.tolist())
    batch_id = None

    exp_sets = []
    if config['specific_solvers']:
        print('Solvers Considered:', config['specific_solvers'])
        df_unified, _ = unify_outputs(df, specific_solvers=config['specific_solvers'], log_dir=log_dir, timestamp=timestamp)
    else:
        print('Solvers Considered:', all_solvers)
        df_unified, _ = unify_outputs(df, log_dir=log_dir, timestamp=timestamp)
    exp_sets.append(df_unified)

    if config['batch_run']:
        print('Batch processing')
        requests = collect_requests(exp_sets, config['unifier_models'], config['query_constraints'] if config['query_constraints'] else None, config['query_excludes'] if config['query_excludes'] else None, log_dir=log_dir, timestamp=timestamp)
        if config['sanity_check']:
            print('SANITY CHECK ON')
            pass
        else:
            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            batch_id = make_batch_request(requests, client)
    else:
        print('Individual processing') # the generations are appended to the output file
        model_outfile = os.path.join(log_dir,'indiv-unified-generations.json')

        for model in config['unifier_models']:
            for dft in exp_sets:
                gen_unification(dft, model, model_outfile,  config['query_constraints'] if config['query_constraints'] else None, config['query_excludes'] if config['query_excludes'] else None, config['sanity_check'])

    run_info = {
        'experiment_name': config['experiment_name'],
        'solver_ingredients_file': config['all_ingredients_file'],
        'solvers': config['specific_solvers'] if config['specific_solvers'] else list(all_solvers),
        'batch': config['batch_run'],
        'unifier_models': config['unifier_models'],
        'query_constraints': config['query_constraints'],
        'query_excludes': config['query_excludes'],
        'batch_id': batch_id
    }
    with open(os.path.join(log_dir, f'{timestamp}-run_info.json'), 'w') as fout:
        json.dump(run_info, fout, indent=2)

def unify_outputs(df, specific_solvers=None, custom_id_start=0, log_dir=default_log_dir, timestamp=None):
    dfs = df if specific_solvers is None else df[df['solver'].isin(specific_solvers)]

    concatenated = []
    i = custom_id_start
    for gid, gdf in dfs.groupby('query'):
        ings, source_mapping = concatenate_ingredients(gdf)
        concatenated.append({
            'custom_id': f'q{i}',
            'query': gid,
            'ingredients': ings,
            'source_mapping': source_mapping,
        })
        i += 1
    df2 = pd.DataFrame(concatenated)

    df2.to_json(os.path.join(log_dir, f'{timestamp}-ingredient_reference.json'), indent=4, orient="records")
    return df2, i


def collect_requests(exp_sets, models, query_constraints=None, exclude_constraints=None, log_dir='logs_unification', timestamp=None):
    requests = []
    for df in exp_sets:
        for _, row in df.iterrows():
            if query_constraints is not None and row['query'] not in query_constraints:
                continue

            if exclude_constraints is not None and row['query'] in exclude_constraints:
                continue

            prompt = row[['query', 'ingredients']].to_json()

            for model in models:
                modelname = 'opus' if 'opus' in model else 'sonnet'

                requests.append(Request(
                    custom_id = f"{row['custom_id']}-{modelname}",
                    params=MessageCreateParamsNonStreaming(
                        model=model,
                        max_tokens=9216,
                        system=[
                            {
                                "type": "text",
                                "text": SYSTEM_PROMPT,
                                "cache_control": {
                                    "type": "ephemeral",
                                }
                            },
                        ],
                        thinking={
                            "type": "enabled",
                            "budget_tokens": 3072,
                        },  # Enable extended thinking mode
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ]
                    )
                ))

    with open(os.path.join(log_dir, f'{timestamp}_batch_requests.json'), 'w') as outfile:
        json.dump(requests, outfile, indent=4)

    print('Request Total: {}'.format(len(requests)))

    return requests


def concatenate_ingredients(df):
    temp = []

    seen_solver = []

    for rid,row in df.iterrows():
        ingredients = parse_json(row.generation)
        if row.solver in seen_solver:
            print(f"Skipping Duplicate Entry for \"{row['query']}\": {row.solver}")
            continue
        seen_solver.append(row.solver)

        for ingredient in ingredients:
            ingredient['source'] = row.solver
            temp.append(ingredient)

    random.Random(4).shuffle(temp)

    idx = 0
    concat = []
    source_mapping = []

    for ingredient in temp:
        ingredient['id'] = idx
        source_mapping.append({
            'id': idx,
            'source': ingredient['source'],
        })
        del ingredient['source']
        concat.append(ingredient)
        idx += 1

    return concat, source_mapping


def gen_unification(df, model, model_outfile, query_constraints=None, exclude_constraints=None, sanity_check=True):
    with open(model_outfile, 'a') as fout:
        for _, row in tqdm.tqdm(df[['query','ingredients']].iterrows(), desc='Running Unification'):

            if query_constraints is not None and row['query'] not in query_constraints:
                continue

            if exclude_constraints is not None and row['query'] in exclude_constraints:
                continue

            prompt = row.to_json()

            if not sanity_check:
                client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                response = client.messages.create(
                    model = model,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            # "cache_control": {
                            #     "type": "ephemeral",
                            # }
                        },
                    ],
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 3072,
                    },  # Enable extended thinking mode
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                }
                            ]
                        }
                    ],
                    max_tokens=9216
                )

                fout.write(json.dumps({
                    'query': row['query'],
                    'system': SYSTEM_PROMPT,
                    'prompt': prompt,
                    'model': model,
                    'generation': response.content[1].text,
                    'usage': {'input_tokens': response.usage.input_tokens, 'output_tokens': response.usage.output_tokens},
                    }) + '\n')
            else:
                print(f'{SYSTEM_PROMPT}\n{prompt}\n===================')



if __name__ == "__main__":
    try:
        with open(os.path.join(os.path.dirname(__file__),'config_unify_ingredients.yaml'), 'r') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: 'config_unify_ingredients.yaml' not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML: {exc}")

    main(data)
