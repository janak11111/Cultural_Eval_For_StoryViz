import os
import json
import argparse


def load_jsonl(path):
    data = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["id"]] = item

    return data


def get_score(item):
    """
    Supports either:
        {"score": 4}
    or
        {"jury_rating": 4}
    """
    if "score" in item:
        return item["score"]

    if "jury_rating" in item:
        return item["jury_rating"]

    raise ValueError(f"No score found for id={item['id']}")


def aggregate(args):

    judge_files = {}

    for judge in args.judges:

        filename = (
            f"{args.dataset}_{judge.capitalize()}_"
            f"{args.language}_Level{args.level}.jsonl"
        )

        judge_files[judge] = os.path.join(
            args.judge_predictions_folder,
            args.dataset,
            filename
        )

    judges = {
        judge: load_jsonl(path)
        for judge, path in judge_files.items()
    }

    common_ids = sorted(
        set.intersection(
            *(set(j.keys()) for j in judges.values())
        )
    )

    print(f"Found {len(common_ids)} common samples.")

    with open(args.output, "w", encoding="utf-8") as outfile:

        for sample_id in common_ids:

            scores = []

            for judge in args.judges:
                scores.append(
                    get_score(judges[judge][sample_id])
                )

            if any(score is None for score in scores):
                jury_rating = None
            else:
                jury_rating = round(sum(scores) / len(scores), 2)

            outfile.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "jury_rating": jury_rating
                    },
                    ensure_ascii=False
                )
                + "\n"
            )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True
    )

    parser.add_argument(
        "--language",
        required=True
    )

    parser.add_argument(
        "--judge_predictions_folder",
        required=True
    )

    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        required=True
    )

    parser.add_argument(
        "--judges",
        nargs="+",
        default=["gemini", "qwen", "maya"]
    )

    parser.add_argument(
        "--output",
        required=True
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    aggregate(args)
