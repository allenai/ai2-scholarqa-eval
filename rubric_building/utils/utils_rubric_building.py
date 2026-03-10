import json


def parse_json(json_str, expected_first_char="[", question=None):
    json_str = json_str.strip()

    if not json_str.startswith(expected_first_char) and not json_str.startswith("```"):
        idxt = json_str.find("```")
        idxb = json_str.find(expected_first_char)

        if idxt > -1:
            json_str = json_str[idxt:]
        else:
            json_str = json_str[idxb:]

    json_str = json_str.strip()

    if not json_str.endswith("```"):
        return json.loads(json_str.lstrip("```json").split("```")[0])
    else:
        return json.loads(json_str.rstrip("```").lstrip("```json"))


def make_batch_request(requests, client):
    response = client.messages.batches.create(requests=requests)
    print(response)
    return response.id
