import argparse
import logging
import os
import pathlib
import sys

import coastal_dynamics as cd


def process_questions(questions):
    """Process and hash answers within the questions dictionary based on question type."""
    for _, q_data in questions.items():
        q_data["answer"] = cd.hash_answer(
            q_data.get("answer"), q_data.get("type"), sig_figs=q_data.get("sig_figs")
        )
    return questions


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process questions from cloud storage."
    )
    parser.add_argument(
        "fname", type=str, help="The file for the questions file to process"
    )
    return parser.parse_args()


def process_file(storage_options, blob_name):
    """Process a single file."""
    fstem = pathlib.Path(blob_name).stem

    questions = cd.read_questions(blob_name)
    processed_questions = process_questions(questions)

    # this blob name can be used to write the hashed questions to local device,
    # but can be updated in future to write to hub?
    hashed_blob_name = str(
        os.path.join(os.getcwd(), f"../hashed_questions/{fstem}_hashed.json")
    )

    cd.write_questions(
        processed_questions,
        hashed_blob_name,
        storage_options=storage_options,
        storage="local",
    )


def main():
    """Main function to orchestrate the processing of questions."""
    args = parse_arguments()
    storage_options = None

    # this blob name can be used to write the hashed questions to local device,
    # but can be updated in future to write to hub?
    blob_name = str(
        # os.path.join(
        #     os.getcwd(),
        #     pathlib.Path(f"../CoastalCodebookSecrets/questions/{pathlib.Path(args.fname).stem}.json")
        # )
        pathlib.Path(args.fname)
    )

    try:
        process_file(storage_options, blob_name)
        logging.info("Questions processed and stored successfully.")
    except Exception as e:
        logging.error(f"Failed to process questions: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
