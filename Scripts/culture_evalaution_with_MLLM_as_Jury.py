import json
import argparse


def load_jsonl(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["id"]] = item

    return data


def aggregate(args):

    judge1 = load_jsonl(args.judge1)
    judge2 = load_jsonl(args.judge2)
    judge3 = load_jsonl(args.judge3)

    common_ids = sorted(
        set(judge1.keys()) &
        set(judge2.keys()) &
        set(judge3.keys())
    )

    print(f"Found {len(common_ids)} common samples.")

    with open(args.output, "w", encoding="utf-8") as outfile:

        for sample_id in common_ids:

            scores = [
                judge1[sample_id]["score"],
                judge2[sample_id]["score"],
                judge3[sample_id]["score"],
            ]

            if any(score is None for score in scores):
                jury_rating = None
            else:
                jury_rating = round(sum(scores) / 3, 2)

            output = {
                "id": sample_id,
                "jury_rating": jury_rating
            }

            outfile.write(
                json.dumps(output, ensure_ascii=False) + "\n"
            )


def parse_args():

    parser = argparse.ArgumentParser(
        description="Aggregate MLLM-as-Judge scores."
    )

    parser.add_argument(
        "--judge1",
        required=True,
        help="Judge 1 JSONL file"
    )

    parser.add_argument(
        "--judge2",
        required=True,
        help="Judge 2 JSONL file"
    )

    parser.add_argument(
        "--judge3",
        required=True,
        help="Judge 3 JSONL file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    aggregate(args)
