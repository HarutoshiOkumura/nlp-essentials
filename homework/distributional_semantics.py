import numpy as np
from typing import Dict, List, Tuple

# GPT is used to comment and check code integrity + improvement -> Baseline is human written + reference to lecture slides


# ---------------------------------------------------------------------------
# Task 1 (Checked)
# ---------------------------------------------------------------------------

def read_word_embeddings(file_path: str) -> Dict[str, np.ndarray]:
    """Read 50‑d Word2Vec embeddings from *file_path*.

    Each line **must** be: ``WORD\tFLOAT FLOAT … (50 total)``.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from *word* → 50‑d NumPy vector (dtype ``float32``).
    """

    embeddings: Dict[str, np.ndarray] = {}
    with open(file_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            parts = line.strip().split()
            if not parts:
                continue  # skip empty lines
            if len(parts) != 51:  # 1 word + 50 numbers
                raise ValueError(
                    f"Line {lineno}: expected 51 tokens (word + 50 dims) but got {len(parts)}"
                )
            word, *vector_str = parts
            vector = np.asarray(vector_str, dtype=np.float32)
            embeddings[word] = vector
    return embeddings


# ---------------------------------------------------------------------------
# Helper Function – Basic Implementation of Cosine Similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return **cosine similarity** between two vectors.
    Returns 0.0 if either vector has zero magnitude."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ---------------------------------------------------------------------------
# Task 2 (Checked -> Not too sure if its completely correct, come back to it later)
# ---------------------------------------------------------------------------

def similar_words(
    embeddings: Dict[str, np.ndarray],
    target_word: str,
    threshold: float,
) -> List[Tuple[str, float]]:
    """Return words whose cosine similarity with *target_word* ≥ *threshold*.
    """

    if target_word not in embeddings:
        raise KeyError(f"Target word '{target_word}' not in the embedding vocabulary. You messed up somewhere haru :3")

    target_vec = embeddings[target_word]
    results: List[Tuple[str, float]] = []
    for word, vec in embeddings.items():
        if word == target_word:
            continue
        sim = _cosine_similarity(target_vec, vec)
        if sim >= threshold:
            results.append((word, sim))

    results.sort(key=lambda pair: pair[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Task 3 (Checked)
# ---------------------------------------------------------------------------

def document_similarity(
    embeddings: Dict[str, np.ndarray],
    doc1: List[str],
    doc2: List[str],
) -> float:
    """Return cosine similarity between two tokenised documents.

    Each document is embedded by **mean pooling** its in‑vocabulary word
    vectors – an application of the *distributional hypothesis* at the
    document level (lecture: "document embedding = average of word
    vectors").
    """

    def _avg_embedding(tokens: List[str]) -> np.ndarray:
        vecs = [embeddings[t] for t in tokens if t in embeddings]
        if not vecs:  # no overlap with vocabulary → zero vector
            return np.zeros(next(iter(embeddings.values())).shape, dtype=np.float32)
        return np.mean(vecs, axis=0)

    vec1 = _avg_embedding(doc1)
    vec2 = _avg_embedding(doc2)
    return _cosine_similarity(vec1, vec2)


# ---------------------------------------------------------------------------
# Quick self‑test (can be removed/commented in production)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # print("Running self‑tests …")

    # toy_embeds = {
    #     "king": np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32),
    #     "man": np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32),
    #     "woman": np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32),
    #     "queen": np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float32),
    # }

    # # ---------- Test 1: similar_words ----------
    # sim_049 = similar_words(toy_embeds, "king", 0.49)
    # sim_051 = similar_words(toy_embeds, "king", 0.51)
    # print("similar_words threshold 0.49 →", sim_049)
    # print("similar_words threshold 0.51 →", sim_051)
    # # Check if queen and man are present with similarity approximately 0.5
    # sim_049_dict = dict(sim_049)
    # queen_present = "queen" in sim_049_dict and abs(sim_049_dict["queen"] - 0.5) < 1e-6
    # man_present = "man" in sim_049_dict and abs(sim_049_dict["man"] - 0.5) < 1e-6
    # assert queen_present and man_present, "T1a failed: expected queen & man with sim≈0.5 for threshold ≥0.49"
    # # Check if queen is present (approx 0.5) and man is absent for threshold 0.51 -> Corrected: both should be absent
    # sim_051_dict = dict(sim_051)
    # queen_absent_051 = "queen" not in sim_051_dict
    # man_absent_051 = "man" not in sim_051_dict
    # assert queen_absent_051 and man_absent_051, "T1b failed: expected neither queen nor man for threshold ≥0.51"

    # # ---------- Test 2: symmetry of document_similarity ----------
    # d1 = ["king", "man"]
    # d2 = ["queen", "woman"]
    # s12 = document_similarity(toy_embeds, d1, d2)
    # s21 = document_similarity(toy_embeds, d2, d1)
    # print("doc sim d1→d2:", s12)
    # assert abs(s12 - s21) < 1e-6, "T2 failed: similarity should be symmetric"

    # # ---------- Test 3: identical documents ----------
    # s11 = document_similarity(toy_embeds, d1, d1)
    # print("doc sim identical (d1,d1):", s11)
    # assert abs(s11 - 1.0) < 1e-6, "T3 failed: identical docs should have sim≈1"

    # # ---------- Test 4: OOV handling ----------
    # d3 = ["unk1", "unk2"]
    # s1oov = document_similarity(toy_embeds, d1, d3)
    # print("doc sim with OOV doc:", s1oov)
    # assert s1oov == 0.0, "T4 failed: sim with zero‑vector doc should be 0"

    print("All self‑tests passed! ✅")
