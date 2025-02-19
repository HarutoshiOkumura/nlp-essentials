from text_processing import load_lexica
from text_processing import process_file
from text_processing import run_tests
from collections import Counter
from text_processing import regular_expressions
import os
import json
import logging


def main():
    delimiters = set('''!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~''')
    # Load lexica from the "sentiment_treebank" directory.
    lexica = load_lexica("sentiment_treebank")
    file_path = os.path.join("dat", "chronicles_of_narnia.txt")
    books_data = process_file(file_path, lexica, delimiters)

    # Write the output dictionary to a JSON file in the same directory.
    output_path = os.path.join(os.path.dirname(__file__), "narinia_masterbook_output.json")
    try:
        with open(output_path, 'w', encoding='utf-8') as fout:
            json.dump(books_data, fout, indent=2)
        logging.info(f"Output successfully written to {output_path}")
    except Exception as e:
        logging.error(f"Error writing output JSON: {e}")

    # Print a summary of results.
    for book_title, book in books_data.items():
        print(f"Book: {book_title} ({book['year']})")
        for chapter in book["chapters"]:
            print(f"  Chapter {chapter['number']}: {chapter['title']} - Tokens: {chapter['token_count']}")

    run_tests(delimiters, lexica)

    """
    ==============================================================================
    Regular Expressions
    ==============================================================================
    """
    regex_tests = [
        'student@emory.edu',  # email
        '2024/12/25',  # date
        'http://www.emory.edu',  # url
        'Smith, 2023'  # citation
    ]
    for test in regex_tests:
        print(f"{test} -> {regular_expressions(test)}")


if __name__ == "__main__":
    main()
