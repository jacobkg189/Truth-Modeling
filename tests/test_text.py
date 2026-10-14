from factchecker.text import (
    bm25_scores,
    entity_phrases,
    chunk_document,
    extract_numbers,
    keyword_query,
    split_sentences,
    tokenize,
)


def test_split_sentences_basic():
    assert split_sentences("Paris is in France. Rome is in Italy! Is Bern in Spain?") == [
        "Paris is in France.",
        "Rome is in Italy!",
        "Is Bern in Spain?",
    ]


def test_split_sentences_keeps_abbreviations_and_initials():
    text = "The U.S. President met Dr. Smith in Washington. J. K. Rowling wrote it."
    assert split_sentences(text) == [
        "The U.S. President met Dr. Smith in Washington.",
        "J. K. Rowling wrote it.",
    ]


def test_split_sentences_handles_closing_quotes_and_whitespace():
    assert split_sentences('He said "stop."  Then\nhe left.') == ['He said "stop."', "Then he left."]
    assert split_sentences("   ") == []


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The towers of Paris were built") == ["tower", "pari", "built"]
    assert tokenize("headquartered headquarters founding founded") == [
        "headquarter", "headquarter", "found", "found",
    ]


def test_bm25_ranks_matching_document_first():
    docs = ["Rome is the capital of Italy.", "The Eiffel Tower is in Paris.", "Bananas are yellow."]
    scores = bm25_scores("Where is the Eiffel Tower?", docs)
    assert scores.index(max(scores)) == 1
    assert scores[2] == 0
    assert bm25_scores("anything", []) == []


def test_extract_numbers_normalizes():
    assert extract_numbers("It is 330 m (1,083 ft) tall, built 1887–1889, ratio 2.50") == {
        "330", "1083", "1887", "1889", "2.5",
    }
    assert extract_numbers("version v2.0 of x9") == set()


def test_keyword_query_keeps_content_words():
    assert keyword_query("Was the Eiffel Tower built in 1889?") == "Eiffel Tower built 1889"


def test_chunk_document_sections_and_reference_cutoff():
    doc = (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.\n"
        "== History ==\n"
        "Construction of the tower began in 1887 and finished in 1889 for the fair.\n"
        "Short caption\n"
        "=== Design ===\n"
        "The design was by Maurice Koechlin and Emile Nouguier, two engineers.\n"
        "== References ==\n"
        "Some citation text that should never become a passage at all.\n"
    )
    chunks = chunk_document(doc)
    assert [section for section, _ in chunks] == [None, "History", "Design"]
    assert all("citation" not in text for _, text in chunks)


def test_chunk_document_windows_long_paragraphs_with_overlap():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(30)]
    chunks = chunk_document(" ".join(sentences), max_words=40)
    assert len(chunks) > 1
    assert all(len(text.split()) <= 48 for _, text in chunks)
    # consecutive windows share one sentence so no fact is cut in half
    first, second = chunks[0][1], chunks[1][1]
    assert split_sentences(first)[-1] == split_sentences(second)[0]


def test_entity_phrases():
    assert entity_phrases("The Great Wall of China is located in Japan.") == ["Great Wall of China", "Japan"]
    assert entity_phrases("Leonardo da Vinci painted the Mona Lisa.") == ["Leonardo da Vinci", "Mona Lisa"]
    assert entity_phrases("Gustave Eiffel's favorite color was green.") == ["Gustave Eiffel"]
    assert entity_phrases("the sky is blue") == []
    assert entity_phrases("Was the Eiffel Tower built in 1889?") == ["Eiffel Tower"]
    assert entity_phrases("In 2004, Facebook launched.") == ["Facebook"]


def test_entity_phrases_titles_with_joiners_and_qualifiers():
    # Full form first, then the pieces, so the lookup works whichever is the real title.
    assert entity_phrases("Cheese in the Trap (TV series) stars Batman.")[:3] == [
        "Cheese in the Trap (TV series)", "Batman", "Cheese in the Trap",
    ]
    assert entity_phrases("Paris in France is nice.") == ["Paris in France", "France", "Paris"]
