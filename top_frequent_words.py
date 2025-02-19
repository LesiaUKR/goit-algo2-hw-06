import string
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import logging
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ColorfulFormatter(logging.Formatter):
    """
    Custom logging formatter that adds color to log messages
    based on their severity level.
    """
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        """
        Format log messages with corresponding color.
        """
        color = self.COLORS.get(record.levelno, "")
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


# Logger setup
handler = logging.StreamHandler()
formatter = ColorfulFormatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger()


def fetch_text(url):
    """
    Fetch text content from a given URL.

    Args:
        url (str): The URL to fetch text from.

    Returns:
        str: The text content if successful, otherwise None.
    """
    try:
        logger.info("Downloading text from URL...")
        response = requests.get(url)
        response.raise_for_status()
        logger.info("Text successfully downloaded.")
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch text: {e}")
        return None


def remove_punctuation(text):
    """
    Remove punctuation from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without punctuation.
    """
    logger.info("Cleaning text by removing punctuation...")
    return text.translate(str.maketrans('', '', string.punctuation))


def map_function(word):
    """
    Map function to create (word, 1) key-value pairs.

    Args:
        word (str): A word from the text.

    Returns:
        tuple: A key-value pair (word, 1).
    """
    return word, 1


def shuffle_function(mapped_values):
    """
    Shuffle function that groups words together.

    Args:
        mapped_values (list): List of (word, 1) pairs.

    Returns:
        dict_items: Grouped key-value pairs.
    """
    logger.info("Grouping results by keys...")
    shuffled = defaultdict(list)
    for key, value in mapped_values:
        shuffled[key].append(value)
    logger.info("Grouping completed.")
    return shuffled.items()


def reduce_function(key_values):
    """
    Reduce function that counts word occurrences.

    Args:
        key_values (tuple): A key-value pair (word, list of counts).

    Returns:
        tuple: A key-value pair (word, total count).
    """
    key, values = key_values
    return key, sum(values)


def map_reduce(text, top_n=10):
    """
    Perform the MapReduce operation on the given text.

    Args:
        text (str): Input text.
        top_n (int, optional): Number of top frequent words to return. Defaults to 10.

    Returns:
        list: A list of tuples containing the most frequent words and their counts.
    """
    cleaned_text = remove_punctuation(text).lower()
    words = cleaned_text.split()
    logger.info(f"Total words in text: {len(words)}")

    with ThreadPoolExecutor() as executor:
        mapped = list(executor.map(map_function, words))
    logger.info("Map stage completed.")

    shuffled = shuffle_function(mapped)

    with ThreadPoolExecutor() as executor:
        reduced = list(executor.map(reduce_function, shuffled))
    logger.info("Reduce stage completed.")

    sorted_word_counts = sorted(reduced, key=lambda x: x[1], reverse=True)
    logger.info("Sorting completed.")
    return sorted_word_counts[:top_n]


def visualize_top_words(word_counts):
    """
    Visualize the top word frequencies using a bar chart.

    Args:
        word_counts (list): A list of tuples containing words and their frequencies.
    """
    logger.info("Visualizing results...")
    words, counts = zip(*word_counts)
    plt.barh(words, counts, color='skyblue')
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title("Top 10 Most Frequent Words")
    plt.gca().invert_yaxis()
    plt.show()
    logger.info("Visualization completed.")


if __name__ == "__main__":
    """
    Main execution: Fetch text, run MapReduce, and visualize results.
    """
    url = "https://gutenberg.net.au/ebooks01/0100021.txt"
    text = fetch_text(url)
    if text:
        top_words = map_reduce(text, top_n=10)
        logger.info("Top 10 words by frequency:")
        for word, count in top_words:
            logger.info(f"{word}: {count}")
        visualize_top_words(top_words)
    else:
        logger.error("Failed to fetch text for analysis.")
