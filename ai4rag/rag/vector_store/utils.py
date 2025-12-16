from collections import defaultdict

from langchain_core.documents import Document


def merge_window_into_a_document(window: list[Document]) -> Document:
    """
    Merges a list of chunks into a single document.
    If consecutive chunks have intersecting merged_text, the merged_text is merged to avoid duplications.

    Parameters
    ----------
    window : list[Document]
        Ordered list of documents for merging.

    Returns
    -------
    Document
        object that contains the merged merged_text of the window documents.
    """

    def merge_metadata(multiple_metadata: list[dict]) -> dict:
        """
        Merges a list of dictionaries (metadata) into one metadata.
        The keys remain the same but the values are changed into lists of values from all metadata.

        Parameters
        ----------
        multiple_metadata : list[dict]
            List of metadata dictionaries to be merged.

        Returns
        -------
        Single
            Merged dictionary (metadata) with unique values in lists for each key.
        """
        if len(multiple_metadata) == 1:
            return multiple_metadata[0]

        merged_metadata = defaultdict(set)
        for metadata in multiple_metadata:
            for key, value in metadata.items():
                if isinstance(value, list):
                    merged_metadata[key].update(value)
                else:
                    merged_metadata[key].add(value)

        result = {}
        for key, value_set in merged_metadata.items():
            value_list = sorted(value_set)
            if len(value_list) == 1:
                result[key] = value_list[0]
            else:
                result[key] = value_list
        return result

    def get_str2_without_intersecting_text(str1: str, str2: str) -> tuple[str, bool]:
        """
        Finds the intersecting merged_text between the suffix of str1 and the prefix of str2.

        Parameters
        ----------
        str1 : str
            The first string.

        str2 : str
            The second string.

        Returns
        -------
        tuple[str, bool]
            1. str2 without its intersection to str1
            2. whether there was an intersection or not
        """
        # Start checking from the longest possible overlap to the shortest
        for i in range(min(len(str1), len(str2)), 0, -1):
            if str1[-i:] == str2[:i]:
                return str2[i:], True
        return str2, False

    def merge_texts(texts: list[str]) -> str:
        """
        Merges a list of texts into a single text string.
        If consecutive texts have intersecting parts, the text is merged to avoid duplications.

        Parameters
        ----------

        :param texts: ordered list of text strings to be merged
        :type texts: List[str]

        :return: single string that contains the merged text
        :rtype: str
        """
        merged_text = ""
        for text in texts:
            text_to_add, has_intersection = get_str2_without_intersecting_text(
                merged_text, text
            )
            if merged_text and not has_intersection:
                merged_text += " "  # Add a space between non-overlapping texts (chunks)
            merged_text += text_to_add
        return merged_text

    texts = [doc.page_content for doc in window]
    merged_text = merge_texts(texts)

    metadata = [doc.metadata for doc in window]
    merged_metadata = merge_metadata(metadata)

    return Document(page_content=merged_text, metadata=merged_metadata)
