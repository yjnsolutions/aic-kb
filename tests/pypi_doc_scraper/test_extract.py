from aic_kb.pypi_doc_scraper.extract import chunk_text


def test_chunk_text_basic():
    """Test basic text chunking with simple content."""
    text = "This is a test." * 1000  # Create text longer than chunk size
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_text_code_blocks():
    """Test chunking with code blocks."""
    text = (
        """
    Here is some text before the code.
    ```python
    def example():
        print("This is a code block")
        return True
    ```
    Here is some text after the code.
    """
        * 10
    )
    chunks = chunk_text(text, chunk_size=200)

    # Check that code blocks aren't split in the middle
    assert not any("```" in chunk and chunk.count("```") == 1 for chunk in chunks)


def test_chunk_text_paragraphs():
    """Test chunking with paragraphs."""
    text = (
        """
    This is paragraph one.
    It has multiple lines.

    This is paragraph two.
    It also has multiple lines.

    This is paragraph three.
    """
        * 10
    )
    chunks = chunk_text(text, chunk_size=100)

    # Verify that chunks do not contain internal paragraph breaks
    assert all("\n\n" not in chunk for chunk in chunks), "Chunks should not contain paragraph breaks within them"


def test_chunk_text_sentences():
    """Test chunking at sentence boundaries."""
    text = "This is sentence one. This is sentence two. This is sentence three. " * 50
    chunks = chunk_text(text, chunk_size=100)

    # Most chunks should end with periods
    assert all(chunk.endswith(".") for chunk in chunks[:-1])


def test_chunk_text_small_input():
    """Test chunking with input smaller than chunk size."""
    text = "This is a small piece of text."
    chunks = chunk_text(text, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty_input():
    """Test chunking with empty input."""
    text = ""
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) == 0


def test_chunk_text_whitespace():
    """Test chunking handles whitespace properly."""
    text = "   Line with spaces   \n\n   Another line   \n\n   Third line   "
    chunks = chunk_text(text, chunk_size=100)
    assert all(chunk == chunk.strip() for chunk in chunks)
