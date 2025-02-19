import re
import os
import json
import logging
from collections import Counter  # Removed usage in final output
from typing import List, Dict, Any, Set, Union

# Configure logging for robust error handling and debugging.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Utility Functions ---

def roman_to_int(roman: str) -> int:
    """Convert a Roman numeral to an integer."""
    roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                      'C': 100, 'D': 500, 'M': 1000}
    result = 0
    prev_value = 0
    for char in reversed(roman):
        value = roman_numerals.get(char, 0)
        if value < prev_value:
            result -= value
        else:
            result += value
            prev_value = value
    return result

# --- Improvement Area 2: Iterative Tokenization ---
def iterative_delimit(word: str, delimiters: Set[str]) -> List[str]:
    """
    Iteratively splits a word based on a set of delimiters.
    This avoids recursion overhead and potential recursion limits.
    """
    tokens = []
    current = ""
    for char in word:
        if char in delimiters:
            if current:
                tokens.append(current)
                current = ""
            tokens.append(char)
        else:
            current += char
    if current:
        tokens.append(current)
    return tokens

def postprocess(tokens: List[str]) -> List[str]:
    """
    Post-process tokens to handle special cases:
    - Combine apostrophes with a following 's'
    - Merge numeric tokens with delimiters for reference numbers or comma-separated numbers
    - Merge tokens for acronyms like "U.S."
    """
    i = 0
    new_tokens = []
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == "'" and tokens[i + 1].lower() == "s":
            new_tokens.append("".join(tokens[i:i+2]))
            i += 2
            continue
        elif i + 2 < len(tokens) and (
            (tokens[i] == '[' and tokens[i+1].isnumeric() and tokens[i+2] == ']') or
            (tokens[i].isnumeric() and tokens[i+1] == ',' and tokens[i+2].isnumeric())
        ):
            new_tokens.append("".join(tokens[i:i+3]))
            i += 3
            continue
        elif i + 3 < len(tokens) and "".join(tokens[i:i+4]) == "U.S.":
            new_tokens.append("".join(tokens[i:i+4]))
            i += 4
            continue
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

def tokenize_text(text: str, delimiters: Set[str]) -> List[str]:
    """
    Tokenizes text by splitting on whitespace, then applying iterative_delimit
    and postprocessing.
    """
    words = text.split()
    tokens = []
    for word in words:
        tokens.extend(postprocess(iterative_delimit(word, delimiters)))
    return tokens

# --- Improvement Area 3: Lexical Resources and Lemmatization ---
def load_lexica(res_dir: str) -> Dict[str, Any]:
    """
    Loads lexical resources for lemmatization from the provided directory.
    Uses try/except blocks for robust file handling.
    """
    lexica = {}
    try:
        with open(os.path.join(res_dir, 'nouns.txt'), 'r') as fin:
            lexica['nouns'] = {line.strip() for line in fin}
    except Exception as e:
        logging.error(f"Error loading nouns.txt: {e}")
        lexica['nouns'] = set()

    try:
        with open(os.path.join(res_dir, 'verbs.txt'), 'r') as fin:
            lexica['verbs'] = {line.strip() for line in fin}
    except Exception as e:
        logging.error(f"Error loading verbs.txt: {e}")
        lexica['verbs'] = set()

    try:
        with open(os.path.join(res_dir, 'nouns_irregular.json'), 'r') as fin:
            lexica['nouns_irregular'] = json.load(fin)
    except Exception as e:
        logging.error(f"Error loading nouns_irregular.json: {e}")
        lexica['nouns_irregular'] = {}

    try:
        with open(os.path.join(res_dir, 'verbs_irregular.json'), 'r') as fin:
            lexica['verbs_irregular'] = json.load(fin)
    except Exception as e:
        logging.error(f"Error loading verbs_irregular.json: {e}")
        lexica['verbs_irregular'] = {}

    try:
        with open(os.path.join(res_dir, 'nouns_rules.json'), 'r') as fin:
            lexica['nouns_rules'] = json.load(fin)
    except Exception as e:
        logging.error(f"Error loading nouns_rules.json: {e}")
        lexica['nouns_rules'] = []

    try:
        with open(os.path.join(res_dir, 'verbs_rules.json'), 'r') as fin:
            lexica['verbs_rules'] = json.load(fin)
    except Exception as e:
        logging.error(f"Error loading verbs_rules.json: {e}")
        lexica['verbs_rules'] = []

    return lexica

def aux_lemmatize(word: str, vocabs: Set[str], irregular: Dict[str, str], rules: List[List[str]]) -> Union[str, None]:
    """
    Attempts to lemmatize a word using irregular forms first and then applying rules.
    """
    if word in irregular:
        return irregular[word]
    for rule in rules:
        pattern, replacement = rule
        if word.endswith(pattern):
            lemma_candidate = word[:-len(pattern)] + replacement
            if lemma_candidate in vocabs:
                return lemma_candidate
    return None

def lemmatize(lexica: Dict[str, Any], word: str) -> str:
    """
    Converts a word to its lemma using provided lexical resources.
    First attempts verb lemmatization, then noun lemmatization.
    (For production, consider established libraries like spaCy.)
    """
    word_lower = word.lower()
    lemma = aux_lemmatize(word_lower, lexica['verbs'], lexica['verbs_irregular'], lexica['verbs_rules'])
    if lemma is None:
        lemma = aux_lemmatize(word_lower, lexica['nouns'], lexica['nouns_irregular'], lexica['nouns_rules'])
    return lemma if lemma else word_lower

# --- Processing Chapters and Books ---
def process_chapter(chapter_text: str, lexica: Dict[str, Any], delimiters: Set[str]) -> Dict[str, Any]:
    """
    Processes a chapter's text by tokenizing, lemmatizing, and computing token count.
    """
    tokens = tokenize_text(chapter_text, delimiters)
    lemmatized_tokens = [lemmatize(lexica, token) for token in tokens]
    total_tokens = len(lemmatized_tokens)
    return {"token_count": total_tokens}

def process_file(file_path: str, lexica: Dict[str, Any], delimiters: Set[str]) -> Dict[str, Any]:
    """
    Processes the Chronicles of Narnia file line-by-line (for memory efficiency)
    to extract books and chapters, then applies the NLP pipeline on each chapter.
    """
    books = {}
    current_book = None
    current_chapter = None
    chapter_text = ""

    # Updated regex for book titles: allow extra whitespace.
    book_title_re = re.compile(r"^(.*?)\s*\(\s*(\d{4})\s*\)\s*$")
    # Updated regex for chapter headers: chapter title is optional on the same line.
    chapter_re = re.compile(r"^CHAPTER\s+([IVXLCDM]+)(?:\s+(.*))?$")

    try:
        with open(file_path, 'r', encoding='utf-8') as fin:
            lines = iter(fin.readlines())
            for line in lines:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                # Check for a book title line.
                title_match = book_title_re.match(line)
                if title_match:
                    current_book = title_match.group(1)
                    year = int(title_match.group(2))
                    books[current_book] = {"title": current_book, "year": year, "chapters": []}
                    logging.info(f"Detected book: {current_book} ({year})")
                    continue

                # Check for a chapter header.
                chapter_match = chapter_re.match(line)
                if chapter_match:
                    # If a chapter is already in progress, process and store it.
                    if current_chapter is not None:
                        chapter_data = process_chapter(chapter_text, lexica, delimiters)
                        current_chapter["token_count"] = chapter_data["token_count"]
                        books[current_book]["chapters"].append(current_chapter)
                        logging.info(f"Processed Chapter {current_chapter['number']} - {current_chapter['title']}")
                    roman = chapter_match.group(1)
                    chapter_number = roman_to_int(roman)
                    # If chapter title is missing on the same line, take the next line.
                    chapter_title = chapter_match.group(2)
                    if chapter_title is None:
                        try:
                            chapter_title = next(lines).strip()
                        except StopIteration:
                            chapter_title = ""
                    current_chapter = {"number": chapter_number, "title": chapter_title}
                    chapter_text = ""
                    continue

                # Accumulate chapter text if inside a chapter.
                if current_chapter is not None:
                    chapter_text += " " + line
                else:
                    logging.debug(f"Line outside any chapter/book: {line}")

            # Process the final chapter at EOF.
            if current_chapter is not None:
                chapter_data = process_chapter(chapter_text, lexica, delimiters)
                current_chapter["token_count"] = chapter_data["token_count"]
                books[current_book]["chapters"].append(current_chapter)
                logging.info(f"Processed final Chapter {current_chapter['number']} - {current_chapter['title']}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

    # Sort chapters for each book by chapter number.
    for book in books.values():
        book["chapters"].sort(key=lambda ch: ch["number"])
    return books

# --- Improvement Area 4: Modularization and Testing ---
def run_tests(delimiters: Set[str], lexica: Dict[str, Any]) -> None:
    """
    Runs simple tests for tokenization and lemmatization.
    """
    sample_text = 'Department\'s activity"[26] centers.[21][22] U.S.'
    tokens = tokenize_text(sample_text, delimiters)
    logging.info(f"Tokenization Test:\nInput: {sample_text}\nOutput: {tokens}")

    sample_words = ['studies', 'crosses', 'children', 'bought']
    lemmatized = [lemmatize(lexica, word) for word in sample_words]
    logging.info(f"Lemmatization Test:\nInput: {sample_words}\nOutput: {lemmatized}")






"""
================================================================================
Regular Expressions
================================================================================
"""
def regular_expressions(text: str) -> Union[str, None]:
    """
    Identifies the type of a given text pattern.

    :param text: String to classify.
    :return: One of "email", "date", "url", "cite"; None if no pattern matches.
    """
    # --- 1. Email ---
    # Username and hostname: start and end with letter/number,
    # can contain letters, digits, period, underscore, or hyphen.
    # Domain must be one of: com, org, edu, gov.
    email_pattern = re.compile(
        r'^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?@'
        r'[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.(?:com|org|edu|gov)$'
    )
    if email_pattern.match(text):
        return "email"

    # --- 2. Date ---
    # Acceptable formats: YYYY/MM/DD, YY/MM/DD, YYYY-MM-DD, YY-MM-DD
    # We capture year, month, and day.
    date_pattern = re.compile(r'^(\d{2}|\d{4})[/-](\d{1,2})[/-](\d{1,2})$')
    m = date_pattern.match(text)
    if m:
        year_str, month_str, day_str = m.groups()
        # Convert 2-digit year to 4-digit.
        if len(year_str) == 2:
            year_val = int(year_str)
            # if year between 51 and 99, assume 1900 + year; if between 00 and 50, assume 2000 + year.
            if 51 <= year_val <= 99:
                year = 1900 + year_val
            else:
                year = 2000 + year_val
        else:
            year = int(year_str)
        # Check year range.
        if not (1951 <= year <= 2050):
            pass  # not a valid date for our criteria.
        else:
            month = int(month_str)
            day = int(day_str)
            # Basic check for month.
            if 1 <= month <= 12:
                # Maximum days per month (allowing February 29 as a maximum).
                max_days = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
                if 1 <= day <= max_days.get(month, 31):
                    return "date"

    # --- 3. URL ---
    # Only "http" or "https" protocols, address must start with letter/number and include at least one dot.
    url_pattern = re.compile(
        r'^(https?)://'
        r'[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)+$'
    )
    if url_pattern.match(text):
        return "url"

    # --- 4. Citation ("cite") ---
    # Three possible formats:
    #   Single author: Lastname, YYYY
    #   Two authors: Lastname and Lastname, YYYY
    #   Multiple authors: Lastname et al., YYYY
    # Lastnames must be capitalized; year must be between 1900 and 2024.
    cite_pattern_single = re.compile(r'^([A-Z][a-zA-Z]+),\s*(\d{4})$')
    cite_pattern_two = re.compile(r'^([A-Z][a-zA-Z]+)\s+and\s+([A-Z][a-zA-Z]+),\s*(\d{4})$')
    cite_pattern_multi = re.compile(r'^([A-Z][a-zA-Z]+)\s+et\s+al\.,\s*(\d{4})$')

    for pattern in (cite_pattern_single, cite_pattern_two, cite_pattern_multi):
        m = pattern.match(text)
        if m:
            # The last captured group is the year.
            year = int(m.groups()[-1])
            if 1900 <= year <= 2024:
                return "cite"

    return None
