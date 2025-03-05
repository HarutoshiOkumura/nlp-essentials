from collections import Counter, defaultdict
import math
import string

# Special tokens.
UNKNOWN = ''
INIT = '[INIT]'

# Type alias for Bigram: dictionary mapping a previous word (str) to a dictionary of current words and probabilities.
Bigram = dict


def bigram_count(filepath: str) -> dict[str, Counter]:
    bigrams = defaultdict(Counter)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            tokens = [INIT] + words  # Prepend the INIT token.
            for i in range(1, len(tokens)):
                bigrams[tokens[i - 1]].update([tokens[i]])
    return bigrams


def bigram_estimation(filepath: str) -> Bigram:
    counts = bigram_count(filepath)
    bigrams = {}
    for prev, ccs in counts.items():
        total = sum(ccs.values())
        bigrams[prev] = {curr: count / total for curr, count in ccs.items()}
    return bigrams


def bigram_smoothing(filepath: str) -> Bigram:
    counts = bigram_count(filepath)
    # Build vocabulary from both previous words and the words following them.
    vocab = set(counts.keys())
    for css in counts.values():
        vocab.update(css.keys())

    bigrams = dict()
    for prev, ccs in counts.items():
        total = sum(ccs.values()) + len(vocab)  # add-one smoothing adjustment.
        # For every token in the vocabulary, if it wasn't seen after 'prev', count is assumed 0.
        d = {curr: (ccs.get(curr, 0) + 1) / total for curr in vocab}
        # Optionally, ensure the unknown token has a defined probability.
        d[UNKNOWN] = 1 / total
        bigrams[prev] = d

    # For unseen previous words, assign a default probability.
    bigrams[UNKNOWN] = 1 / len(vocab)
    return bigrams



# ==============================================================================
#  Sequence Generation
# ==============================================================================





def is_punctuation(token: str) -> bool:

    return token in string.punctuation


def sequence_generator(bigram_model: dict, initial_word: str, sequence_length: int) -> tuple:
    # Maximum allowed punctuation tokens.
    allowed_punc = sequence_length // 5  # floor division
    sequence = [initial_word]
    log_likelihood = 0.0
    punctuation_count = 0
    non_punc_set = set()

    # Update the set/counters for the initial token.
    if is_punctuation(initial_word):
        punctuation_count += 1
    else:
        non_punc_set.add(initial_word)

    current_word = initial_word

    # Generate tokens until the desired sequence length is reached.
    while len(sequence) < sequence_length:
        # Get candidate next tokens from the bigram model.
        # If current word is unseen, use the UNKNOWN fallback.
        if current_word not in bigram_model:
            candidates = [('', bigram_model.get('', 1e-12))]
        else:
            # Sort candidates by probability (highest first)
            candidates = sorted(bigram_model[current_word].items(), key=lambda x: x[1], reverse=True)

        candidate_found = False
        chosen_token = None
        chosen_prob = None

        # First pass: enforce both punctuation and uniqueness (for non-punctuation).
        for token, prob in candidates:
            if is_punctuation(token):
                if punctuation_count < allowed_punc:
                    chosen_token = token
                    chosen_prob = prob
                    candidate_found = True
                    break
            else:
                if token not in non_punc_set:
                    chosen_token = token
                    chosen_prob = prob
                    candidate_found = True
                    break

        # Fallback: if no candidate meets uniqueness, allow a duplicate (but still enforce punctuation limit).
        if not candidate_found:
            for token, prob in candidates:
                if is_punctuation(token):
                    if punctuation_count < allowed_punc:
                        chosen_token = token
                        chosen_prob = prob
                        candidate_found = True
                        break
                else:
                    chosen_token = token
                    chosen_prob = prob
                    candidate_found = True
                    print(f"Warning: Redundant non-punctuation token '{token}' used.")
                    break

        if not candidate_found or chosen_token is None:
            print("No suitable candidate found. Terminating sequence generation early.")
            break

        # Append the chosen token and update log-likelihood.
        sequence.append(chosen_token)
        # Avoid log(0) by ensuring a minimal probability.
        log_likelihood += math.log(chosen_prob if chosen_prob > 0 else 1e-12)

        # Update punctuation count or non-punctuation set.
        if is_punctuation(chosen_token):
            punctuation_count += 1
        else:
            non_punc_set.add(chosen_token)

        # Update the current word for the next iteration.
        current_word = chosen_token

    return sequence, log_likelihood



# ==============================================================================
#  Sequence Generation PLUS
# ==============================================================================
def sequence_generator_plus(bigram_model: dict, initial_word: str, sequence_length: int, beam_width: int = 3) -> tuple:
    allowed_punc = sequence_length // 5  # maximum allowed punctuation tokens

    # Each candidate in the beam is represented as a tuple:
    # (sequence (list of tokens), log_likelihood, punctuation_count, non_punctuation_set)
    initial_candidate = ([initial_word], 0.0, 1 if is_punctuation(initial_word) else 0,
                         set() if is_punctuation(initial_word) else {initial_word})
    beam = [initial_candidate]

    # Extend sequences until we have the required number of tokens.
    for _ in range(sequence_length - 1):
        new_beam = []
        for seq, log_like, punc_count, non_punc_set in beam:
            current_word = seq[-1]
            # Retrieve next-token candidates for the current word.
            if current_word not in bigram_model:
                candidates = [('', bigram_model.get('', 1e-12))]
            else:
                candidates = sorted(bigram_model[current_word].items(), key=lambda x: x[1], reverse=True)
            # Extend the candidate sequence with each candidate token that meets the constraints.
            for token, prob in candidates:
                # Enforce punctuation limit.
                new_punc_count = punc_count + (1 if is_punctuation(token) else 0)
                if new_punc_count > allowed_punc:
                    continue
                # Enforce uniqueness for non-punctuation tokens.
                if not is_punctuation(token) and token in non_punc_set:
                    continue
                new_seq = seq + [token]
                new_log_like = log_like + math.log(prob if prob > 0 else 1e-12)
                new_non_punc_set = non_punc_set.copy()
                if not is_punctuation(token):
                    new_non_punc_set.add(token)
                new_beam.append((new_seq, new_log_like, new_punc_count, new_non_punc_set))
                # Optionally limit the number of extensions per candidate to reduce search space.
                if len(new_beam) >= beam_width * 5:
                    break
        # If no valid extension was found, retain the current beam.
        if not new_beam:
            new_beam = beam
        # Prune to keep only the top 'beam_width' candidates based on log-likelihood.
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

    # Select the best candidate (highest log-likelihood) from the final beam.
    best_candidate = max(beam, key=lambda x: x[1])
    return best_candidate[0], best_candidate[1]






if __name__ == "__main__":
    corpus = 'dat/chronicles_of_narnia.txt'
    print("\n" + "=" * 50)
    print(f"Building bigram model from file: {corpus}")
    print("=" * 50)
    model = bigram_smoothing(corpus)

    # Sort previous tokens by frequency (descending) and select the top 10.
    print("\n" + "=" * 50)
    print("TOP 10 MOST FREQUENT PREVIOUS WORDS WITH BIGRAM PROBABILITIES")
    print("=" * 50)
    for prev in ['[INIT]', 'I', 'the', 'Aslan', 'Witch', 'Narnia']:
        print("\n-------------------------------------")
        if prev in model:
            print(f"Bigram probabilities for '{prev}':")
            # Sort bigram probabilities for this previous token in descending order.
            sorted_bigrams = sorted(model[prev].items(), key=lambda x: x[1], reverse=True)
            for curr, prob in sorted_bigrams[:10]:
                print(f"   {curr:>15} : {prob:.6f}")
        else:
            print("You messed up somewhere Haru...")

    # For unseen previous words, define a fallback probability.
    model[''] = 1e-3

    # Generate a sequence of 10 tokens starting with the initial word 'The'
    seq, log_like = sequence_generator(model, 'The', 10)
    print("Generated Sequence:", seq)
    print("Log-Likelihood:", log_like)


    seq, log_like = sequence_generator_plus(model, 'The', 10)
    print("PLUS Generated Sequence:", seq)
    print("PLUS Log-Likelihood:", log_like)
