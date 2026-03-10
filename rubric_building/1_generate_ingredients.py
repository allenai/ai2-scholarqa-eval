import os
import json
import pandas as pd
import time
import tqdm
import yaml

from pathlib import Path

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from utils.utils_rubric_building import make_batch_request
from utils.prompts import INGREDIENT_EXTRACTION_PROMPT as SYSTEM_PROMPT


default_log_dir = "logs_extraction"


def extract_answer(response):
    if isinstance(response, str):
        return response
    sections = response.get("sections", [])
    parts = []
    for section in sections:
        title = (section.get("title") or "").strip()
        text = (section.get("text") or "").strip()
        if title and text:
            parts.append(f"{title}\n\n{text}")
        elif text:
            parts.append(text)
    return "\n\n".join(parts)


def compile_answers(solver_reports):
    records = []
    for json_file in solver_reports.glob("*.json"):
        solver = json_file.stem
        with open(json_file) as f:
            data = json.load(f)
        for item in data:
            query = item.get("question", "")
            answer = extract_answer(item.get("response", ""))
            records.append({"query": query, "solver": solver, "answer": answer})
    return pd.DataFrame(records)


def main(config: dict):
    solver_reports = Path(config["solver_reports_path"])

    if not os.path.isdir(config["solver_reports_path"]):
        solver_reports = Path(
            os.path.join(os.path.dirname(__file__), config["solver_reports_path"])
        )
        print(solver_reports)
        if not os.path.isdir(solver_reports):
            print(
                f"Exiting the solver_reports_path ({config['solver_reports_path']}) does not exist."
            )
            exit(1)

    log_dir = os.path.join(
        os.path.dirname(__file__),
        config["log_dir"] if config["log_dir"] != "" else default_log_dir,
    )
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M")
    df_input = compile_answers(solver_reports)
    print("Models:", config["extractor_model"])
    all_solvers = set(df_input.solver.tolist())
    batch_id = None

    if config["specific_solvers"]:
        print("Solvers Considered:", config["specific_solvers"])
        df, _ = format_outputs(
            df_input,
            specific_solvers=config["specific_solvers"],
            log_dir=log_dir,
            timestamp=timestamp,
        )
    else:
        print("Solvers Considered:", all_solvers)
        df, _ = format_outputs(df_input, log_dir=log_dir, timestamp=timestamp)

    if config["batch_run"]:
        print("Processing: Batch Run")
        requests = collect_requests(
            df,
            config["extractor_model"],
            config["query_constraints"] if config["query_constraints"] else None,
            config["query_excludes"] if config["query_excludes"] else None,
            log_dir=log_dir,
            timestamp=timestamp,
        )
        if config["sanity_check"]:
            print("SANITY CHECK ON (Log files created; No API call)")
            pass
        else:
            batch_id = make_batch_request(
                requests, client=Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            )
    else:
        print("Processing: Individual Run")
        model_outfile = f"{log_dir}/indiv-extractions.json"

        generate_claude(
            df,
            config["extractor_model"],
            model_outfile,
            config["query_constraints"] if config["query_constraints"] else None,
            config["query_excludes"] if config["query_excludes"] else None,
            config["sanity_check"],
        )

    run_info = {
        "experiment_name": config["experiment_name"],
        "solver_reports_path": config["solver_reports_path"],
        "solvers": config["specific_solvers"]
        if config["specific_solvers"]
        else list(all_solvers),
        "batch": config["batch_run"],
        "unifier_models": config["extractor_model"],
        "query_constraints": config["query_constraints"],
        "query_excludes": config["query_excludes"],
        "batch_id": batch_id,
    }

    with open(os.path.join(log_dir, f"{timestamp}-run_info.json"), "w") as fout:
        json.dump(run_info, fout, indent=2)


def format_outputs(
    df,
    specific_solvers=None,
    custom_id_start=0,
    log_dir=default_log_dir,
    timestamp=None,
):
    dfs = df if specific_solvers is None else df[df["solver"].isin(specific_solvers)]

    responses = []
    i = custom_id_start
    for rid, row in dfs.iterrows():
        responses.append(
            {
                "custom_id": f"q{i}",
                "query": row["query"],
                "solver": row["solver"],
                "report": row["answer"],
                "system": f"{SYSTEM_PROMPT}\n",
            }
        )
        i += 1

    df2 = pd.DataFrame(responses)
    df2.to_json(
        os.path.join(log_dir, f"{timestamp}-request_reference.json"),
        indent=4,
        orient="records",
    )
    return df2, i


def collect_requests(
    df,
    model,
    query_constraints=None,
    exclude_constraints=None,
    log_dir=default_log_dir,
    timestamp=None,
):
    requests = []
    for _, row in df.iterrows():
        if query_constraints is not None and row["query"] not in query_constraints:
            continue

        if exclude_constraints is not None and row["query"] in exclude_constraints:
            continue

        prompt = row[["query", "report"]].to_json()
        requests.append(
            Request(
                custom_id=row["custom_id"],
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=8192,
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
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
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


def generate_claude(
    df,
    model,
    model_outfile,
    query_constraints=None,
    exclude_constraints=None,
    sanity_check=True,
):
    with open(model_outfile, "a") as fout:
        for _, item in tqdm.tqdm(
            df.iterrows(), desc=f"Extracting ingredients from {model}"
        ):
            if query_constraints is not None and item["query"] not in query_constraints:
                continue

            if exclude_constraints is not None and item["query"] in exclude_constraints:
                continue

            prompt = item[["query", "report"]].to_json()

            if not sanity_check:
                client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                response = client.messages.create(
                    model=model,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 3072,
                    },  # Enable extended thinking mode
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                )

                response_out = {
                    "query": item["query"],
                    "solver": item["solver"],
                    "report": item["report"],
                    "system": SYSTEM_PROMPT,
                    "prompt": prompt,
                    "model": model,
                    "generation": response.content[1].text,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                }
                fout.write(json.dumps(response_out) + "\n")
            # else:
            #     print(f'{SYSTEM_PROMPT}\n{prompt}\n===================')


if __name__ == "__main__":
    try:
        with open(
            os.path.join(os.path.dirname(__file__), "config_extract_ingredients.yaml"),
            "r",
        ) as file:
            data = yaml.safe_load(file)
        main(data)
    except FileNotFoundError:
        print("Error: 'config_extract_ingredients.yaml' not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML: {exc}")
