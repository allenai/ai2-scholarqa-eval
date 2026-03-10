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


from utils.utils_rubric_building import parse_json, make_batch_request
from utils.prompts import RUBRIC_UNIFICATION_PROMPT as SYSTEM_PROMPT

default_log_dir = "logs_unification"


def main(config: dict):
    log_dir = os.path.join(
        os.path.dirname(__file__),
        config["log_dir"] if config["log_dir"] != "" else default_log_dir,
    )
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M")

    all_ingredients_file = config["all_ingredients_file"]
    if not os.path.isfile(all_ingredients_file):
        all_ingredients_file = os.path.join(
            os.path.dirname(__file__), config["all_ingredients_file"]
        )

        if not os.path.isfile(all_ingredients_file):
            print(
                f"Exiting the solver_reports_path ({config['all_ingredients_file']}) does not exist."
            )
            exit(1)

    with open(all_ingredients_file) as f:
        df = pd.read_json(f, lines=True)

    print("Models:", config["unifier_models"])
    all_solvers = set(df.solver.tolist())
    batch_id = None

    exp_sets = []
    if config["specific_solvers"]:
        print("Solvers Considered:", config["specific_solvers"])
        df_unified, _ = unify_outputs(
            df,
            specific_solvers=config["specific_solvers"],
            log_dir=log_dir,
            timestamp=timestamp,
        )
    else:
        print("Solvers Considered:", all_solvers)
        df_unified, _ = unify_outputs(df, log_dir=log_dir, timestamp=timestamp)
    exp_sets.append(df_unified)

    if config["batch_run"]:
        print("Processing: Batch Run")
        requests = collect_requests(
            exp_sets,
            config["unifier_models"],
            config["query_constraints"] if config["query_constraints"] else None,
            config["query_excludes"] if config["query_excludes"] else None,
            log_dir=log_dir,
            timestamp=timestamp,
        )
        if config["sanity_check"]:
            print("SANITY CHECK ON (Log files created; No API call)")
            pass
        else:
            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            batch_id = make_batch_request(requests, client)
    else:
        # the generations are appended to the output file
        print("Processing: Individual Run")
        model_outfile = os.path.join(log_dir, "indiv-unified-generations.json")

        for model in config["unifier_models"]:
            for dft in exp_sets:
                gen_unification(
                    dft,
                    model,
                    model_outfile,
                    config["query_constraints"]
                    if config["query_constraints"]
                    else None,
                    config["query_excludes"] if config["query_excludes"] else None,
                    config["sanity_check"],
                )

    run_info = {
        "experiment_name": config["experiment_name"],
        "solver_ingredients_file": config["all_ingredients_file"],
        "solvers": config["specific_solvers"]
        if config["specific_solvers"]
        else list(all_solvers),
        "batch": config["batch_run"],
        "unifier_models": config["unifier_models"],
        "query_constraints": config["query_constraints"],
        "query_excludes": config["query_excludes"],
        "batch_id": batch_id,
    }
    with open(os.path.join(log_dir, f"{timestamp}-run_info.json"), "w") as fout:
        json.dump(run_info, fout, indent=2)


def unify_outputs(
    df,
    specific_solvers=None,
    custom_id_start=0,
    log_dir=default_log_dir,
    timestamp=None,
):
    dfs = df if specific_solvers is None else df[df["solver"].isin(specific_solvers)]

    concatenated = []
    i = custom_id_start
    for gid, gdf in dfs.groupby("query"):
        ings, source_mapping = concatenate_ingredients(gdf)
        concatenated.append(
            {
                "custom_id": f"q{i}",
                "query": gid,
                "ingredients": ings,
                "source_mapping": source_mapping,
            }
        )
        i += 1
    df2 = pd.DataFrame(concatenated)

    df2.to_json(
        os.path.join(log_dir, f"{timestamp}-ingredient_reference.json"),
        indent=4,
        orient="records",
    )
    return df2, i


def collect_requests(
    exp_sets,
    models,
    query_constraints=None,
    exclude_constraints=None,
    log_dir="logs_unification",
    timestamp=None,
):
    requests = []
    for df in exp_sets:
        for _, row in df.iterrows():
            if query_constraints is not None and row["query"] not in query_constraints:
                continue

            if exclude_constraints is not None and row["query"] in exclude_constraints:
                continue

            prompt = row[["query", "ingredients"]].to_json()

            for model in models:
                modelname = "opus" if "opus" in model else "sonnet"

                requests.append(
                    Request(
                        custom_id=f"{row['custom_id']}-{modelname}",
                        params=MessageCreateParamsNonStreaming(
                            model=model,
                            max_tokens=9216,
                            system=[
                                {
                                    "type": "text",
                                    "text": SYSTEM_PROMPT,
                                    "cache_control": {
                                        "type": "ephemeral",
                                    },
                                },
                            ],
                            thinking={
                                "type": "enabled",
                                "budget_tokens": 3072,
                            },  # Enable extended thinking mode
                            messages=[
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": prompt}],
                                }
                            ],
                        ),
                    )
                )

    with open(
        os.path.join(log_dir, f"{timestamp}_batch_requests.json"), "w"
    ) as outfile:
        json.dump(requests, outfile, indent=4)

    print("Request Total: {}".format(len(requests)))

    return requests


def concatenate_ingredients(df):
    temp = []

    seen_solver = []

    for rid, row in df.iterrows():
        ingredients = parse_json(row.generation)
        if row.solver in seen_solver:
            print(f'Skipping Duplicate Entry for "{row["query"]}": {row.solver}')
            continue
        seen_solver.append(row.solver)

        for ingredient in ingredients:
            ingredient["source"] = row.solver
            temp.append(ingredient)

    random.Random(4).shuffle(temp)

    idx = 0
    concat = []
    source_mapping = []

    for ingredient in temp:
        ingredient["id"] = idx
        source_mapping.append(
            {
                "id": idx,
                "source": ingredient["source"],
            }
        )
        del ingredient["source"]
        concat.append(ingredient)
        idx += 1

    return concat, source_mapping


def gen_unification(
    df,
    model,
    model_outfile,
    query_constraints=None,
    exclude_constraints=None,
    sanity_check=True,
):
    with open(model_outfile, "a") as fout:
        for _, row in tqdm.tqdm(
            df[["query", "ingredients"]].iterrows(), desc="Running Unification"
        ):
            if query_constraints is not None and row["query"] not in query_constraints:
                continue

            if exclude_constraints is not None and row["query"] in exclude_constraints:
                continue

            prompt = row.to_json()

            if not sanity_check:
                client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                response = client.messages.create(
                    model=model,
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
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                }
                            ],
                        }
                    ],
                    max_tokens=9216,
                )

                fout.write(
                    json.dumps(
                        {
                            "query": row["query"],
                            "system": SYSTEM_PROMPT,
                            "prompt": prompt,
                            "model": model,
                            "generation": response.content[1].text,
                            "usage": {
                                "input_tokens": response.usage.input_tokens,
                                "output_tokens": response.usage.output_tokens,
                            },
                        }
                    )
                    + "\n"
                )
            else:
                print(f"{SYSTEM_PROMPT}\n{prompt}\n===================")


if __name__ == "__main__":
    try:
        with open(
            os.path.join(os.path.dirname(__file__), "config_unify_ingredients.yaml"),
            "r",
        ) as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: 'config_unify_ingredients.yaml' not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML: {exc}")

    main(data)
