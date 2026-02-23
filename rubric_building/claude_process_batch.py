import argparse
import pandas as pd
import json
import os

from anthropic import Anthropic

from utils_rubric_building import parse_json

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client

def main(this_run, run_prefix, log_dir, saved_generation=None):
    run_info = json.load(open(os.path.join(log_dir, f'{run_prefix}-run_info.json')))
    batch_id = run_info['batch_id']

    if batch_id is None:
        print("No batch ID found")
        exit(1)


    if this_run == 'process_unification_results':

        # converted output
        path_unified = 'outputs/'
        if not os.path.exists(path_unified):
            os.makedirs(path_unified)

        # Load question ID mapping if available (optional)
        question_id_mapping = None
        if os.path.exists('question_id_mapping.json'):
            question_id_mapping = {item['question']: item['case_id'] for item in json.load(open('question_id_mapping.json'))}
        else:
            print("Note: question_id_mapping.json not found. Using sequential IDs.")


        generation_output = os.path.join(log_dir, f'{run_prefix}_generation_output.json')

        unified_prefix = f'{run_prefix}_unified_rubrics'
        ingredient_ref_prefix = f'{run_prefix}-ingredient_reference'

        dfs = []
        for filename in os.listdir(log_dir):
            if filename.startswith(ingredient_ref_prefix):
                print('Processing:', filename)
                dft = pd.read_json(os.path.join(log_dir,filename), orient='records')
                dft['exp'] = run_info['experiment_name']
                dfs.append(dft)

        df = pd.concat(dfs, ignore_index=True)

        if saved_generation:
            dfm = pd.read_json(saved_generation, lines=True)
        else:
            gen = []
            total = 0
            successful = 0
            for result in get_client().messages.batches.results(batch_id):
                total += 1
                if result.result.type == 'succeeded':
                    if len(result.result.message.content) == 0:
                        print(f'Generation failed: {result.custom_id}')
                    else:
                        successful += 1
                        id,model = result.custom_id.split('-')
                        gen.append({
                            'custom_id': id,
                            'model': result.result.message.model,
                            'generation': result.result.message.content[1].text,
                            'thinking': result.result.message.content[0].thinking,
                        })
                elif result.result.type == 'errored':
                    if result.result.error.type == "invalid_request":
                        # Request body must be fixed before re-sending request
                        print(f"Validation error {result.custom_id}")
                        print(result.result.error)
                    else:
                        # Request can be retried directly
                        print(f"Server error {result.custom_id}")
                        print(result.result.error)
                else:
                    print(f"Request {result.result.type} {result.custom_id}")

            print(f'Successful: {successful} / Total: {total}')

            df2 = pd.DataFrame(gen)
            dfm = pd.merge(df2,df, on='custom_id', how='left')
            dfm.to_json(generation_output, orient='records', lines=True)
            print(f'Saved to {generation_output}')

        #df = pd.concat([pd.read_json(open(r), orient='records',lines=True) for r in results], ignore_index=True)
        if len(dfm)>0:
            for expid, gdf_exp in dfm.groupby('exp'):
                out_json_dict = []
                print(f"\nProcessing {expid}")
                outfilename = os.path.join(log_dir, f"{unified_prefix}-{expid}.jsonl")
                query_count = 0
                for gid, gdf in gdf_exp.groupby('query'):
                    unions = []
                    for _, row in gdf.iterrows():
                        try:
                            json_string = parse_json(row.generation, expected_first_char='{', question=gid)
                            unions.append({
                                'model': row.model,
                                'ingredients': json_string['key_requirements'],
                                'left_out_ingredients': json_string['left_out_ingredients']
                            })
                            query_count += 1
                        except:
                            print('==== Couldnt process: ', gid, row.model)

                    out_json_dict.append({
                        'query': gid,
                        'exp': expid,
                        'unions': unions
                    })

                print(f'\tProcessed {query_count} queries')
                print(f'\tWriting unified ingredients to {outfilename}')
                with open(outfilename, 'w') as outfile:
                    json.dump(out_json_dict, outfile, indent=4)

                convert_gen_rubric(out_json_dict, os.path.join(path_unified, f"{run_prefix}_rubrics_{expid}.jsonl"), question_id_mapping)



    elif this_run == 'process_extraction_results':
        # reference file
        reference_file = os.path.join(log_dir, run_prefix + "-request_reference.json")
        df_ref = pd.read_json(reference_file, orient='records')

        outfilename = os.path.join(log_dir,f'{run_prefix}-generation_output.jsonl')

        gen = []
        for result in get_client().messages.batches.results(batch_id):
            if result.result.type == 'succeeded':
                if len(result.result.message.content) == 0:
                    print(f'Generation failed: {result.custom_id}')
                else:
                    gen.append({
                        'custom_id': result.custom_id,
                        'model': result.result.message.model,
                        'generation': result.result.message.content[1].text,
                        'thinking': result.result.message.content[0].thinking,
                    })

        df = pd.merge(pd.DataFrame(gen), df_ref, on='custom_id', how='left')
        df[df.report != ""].to_json(outfilename, orient='records', lines=True)

        print('Writing to extracted outputs to {}'.format(outfilename))

    elif this_run == "check_status":
        check_status(batch_id)

    elif this_run == "cancel_batch":
        message_batch = get_client().messages.batches.cancel(batch_id)
        print(message_batch)


def check_status(batch_id):
    message_batch = get_client().messages.batches.retrieve(batch_id)
    print(f"Batch {message_batch.id} processing status is {message_batch.processing_status}\n")
    print(message_batch)



def convert_gen_rubric(rubrics_data, output_file, question_id_mapping=None):
    converted_data = []
    for idx, rubric_item in enumerate(rubrics_data):
        question = rubric_item['query']
        requirements = rubric_item['unions'][0]['ingredients']
        answer_critical = []
        valuable = []
        for req in requirements:
            modal, criterion = parse_requirement(req['key_requirement'])
            if modal == 'should':
                answer_critical.append({
                    'ingredient': criterion,
                    'examples': [ex['detail'] for ex in req['examples']]
                })
            elif modal == 'might':
                valuable.append({
                    'ingredient': criterion,
                    'examples': [ex['detail'] for ex in req['examples']]
                })
            else:
                print('uh on', req['key_requirement'])

        converted_data.append({
                'question': question,
                'ingredients': calculate_weights(answer_critical, valuable),
                'case_id': question_id_mapping[question] if question_id_mapping is not None and question in question_id_mapping else idx,
                'annotator': 'auto',
                'exp': rubric_item['exp'],
            })

    # converted_data = sorted(converted_data, key=lambda x: int(x["case_id"]))
    # Write the converted data to the output JSON file
    with open(output_file, "w") as outfile:
        json.dump(converted_data, outfile, indent=2)

    print(f"Conversion complete. Output saved to {output_file}")

def parse_requirement(requirement):
    segs = requirement.split(' ')
    modal_idx = 0 if segs[0].lower() in ['should', 'might'] else 2
    modal = segs[modal_idx]
    criterion = segs[modal_idx+1][0].upper() + segs[modal_idx+1][1:] + ' ' + ' '.join(segs[modal_idx+2:])

    return modal.lower(), criterion


# Function to calculate weights and populate other_properties
def calculate_weights(answer_critical, valuable):
    total_critical = len(answer_critical)
    total_valuable = len(valuable)
    total_weight = 2 * total_critical + total_valuable

    other_properties = []
    # Add answer_critical ingredients
    for index, item in enumerate(answer_critical):
        other_properties.append({
            "name": f"answer_critical_{index}",
            "criterion": item["ingredient"],
            "weight": (2 / total_weight),
            "examples": item["examples"]
        })

    # Add valuable ingredients
    for index, item in enumerate(valuable):
        other_properties.append({
            "name": f"valuable_{index}",
            "criterion": item["ingredient"],
            "weight": (1 / total_weight),
            "examples": item["examples"]
        })

    return other_properties

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run", required=True, type=str, help="check_status | process_extraction_results | process_unification_results | cancel_batch")
    parser.add_argument("-l", "--log_dir", type=str, help="log directory (e.g., logs_extraction or logs_unification)", required=True)
    parser.add_argument("-p", "--run_prefix", type=str, help="run prefix, e.g., 20250624-2123 (all files assoc with the same batch should have the same prefix)")
    parser.add_argument("-s", "--saved_generation", type=str, help="filename of saved generation in case you've already pulled this from anthropic already, and don't want to reissue the pull")

    args = parser.parse_args()
    if args.run == "process_results" and args.run_prefix is None:
        parser.error("--run process_results requires --run_prefix")

    else:
        main(args.run, args.run_prefix, args.log_dir, saved_generation=args.saved_generation if args.saved_generation else None)