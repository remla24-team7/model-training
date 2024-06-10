import random
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# pylint: disable=no-name-in-module,import-error
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Example synonyms dictionary for path and query mutations
synonyms_dict = {
    "search": ["find", "lookup"],
    "product": ["item", "goods"],
    "category": ["type", "class"],
    "page": ["p", "pg"],
    "id": ["identifier", "id"],
    "123": ["321", "456"],
    "2": ["two", "dos"],
}


def generate_synonyms(word):
    """Return a list of synonyms for the given word if available, else return the word itself."""
    return synonyms_dict.get(word, [word])


def mutate_protocol(parsed_url):
    """Return a list with a single protocol mutation."""
    protocol = "https" if parsed_url.scheme == "http" else "http"
    return [
        urlunparse(
            (
                protocol,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                parsed_url.query,
                parsed_url.fragment,
            )
        )
    ]


def mutate_domain(parsed_url):
    """Return a list with a single domain mutation."""
    domain_parts = parsed_url.netloc.split(".")
    if "www" in domain_parts:
        domain_parts.remove("www")
    else:
        domain_parts.insert(0, "www")
    domain = ".".join(domain_parts)
    return [
        urlunparse(
            (
                parsed_url.scheme,
                domain,
                parsed_url.path,
                parsed_url.params,
                parsed_url.query,
                parsed_url.fragment,
            )
        )
    ]


def mutate_path(parsed_url):
    """Return a list with path mutations."""
    path_parts = parsed_url.path.strip("/").split("/")
    path_mutants = []
    for i, part in enumerate(path_parts):
        part_synonyms = generate_synonyms(part)
        for synonym in part_synonyms:
            if synonym != part:
                new_path_parts = path_parts[:]
                new_path_parts[i] = synonym
                path_mutants.append("/" + "/".join(new_path_parts))
    return [
        urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                path,
                parsed_url.params,
                parsed_url.query,
                parsed_url.fragment,
            )
        )
        for path in path_mutants
    ]


def mutate_query(parsed_url):
    """Return a list with query parameter mutations."""
    query_params = parse_qs(parsed_url.query)
    query_mutants = []
    for key, values in query_params.items():
        key_synonyms = generate_synonyms(key)
        for new_key in key_synonyms:
            if new_key != key:
                new_query_params = query_params.copy()
                new_query_params[new_key] = new_query_params.pop(key)
                new_query = urlencode(new_query_params, doseq=True)
                query_mutants.append(new_query)
        for value in values:
            value_synonyms = generate_synonyms(value)
            for new_value in value_synonyms:
                if new_value != value:
                    new_query_params = query_params.copy()
                    new_query_params[key] = [new_value]
                    new_query = urlencode(new_query_params, doseq=True)
                    query_mutants.append(new_query)

    return [
        urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                query,
                parsed_url.fragment,
            )
        )
        for query in query_mutants
    ]


def generate_single_feature_mutants(url, num_mutations=5):
    parsed_url = urlparse(url)

    # Collect all possible mutations for each feature
    protocol_mutants = mutate_protocol(parsed_url)
    domain_mutants = mutate_domain(parsed_url)
    path_mutants = mutate_path(parsed_url)
    query_mutants = mutate_query(parsed_url)

    # Combine all mutations into one list
    all_mutants = protocol_mutants + domain_mutants + path_mutants + query_mutants

    # Ensure we only return a specified number of unique mutations
    unique_mutants = list(set(all_mutants))
    num_mutations = max(num_mutations, len(unique_mutants))
    return random.sample(unique_mutants, num_mutations)


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def generate_mutamorphic_sample(url, preprocess):
    """Generate a sample with a mutated URL and its original URL."""
    mutated_urls = generate_single_feature_mutants(
        url, num_mutations=10
    )  # mutate the original URL

    for mutated_url in mutated_urls:  # structural filtreing
        if not is_valid_url(mutated_url):
            mutated_urls.remove(mutated_url)

    preprocessed_urls = preprocess(mutated_urls)  # preprocess the mutated URLs
    preprocessed_original = preprocess([url])

    similarities = cosine_similarity(preprocessed_original, preprocessed_urls)

    # Find the mutated URL that has the highest similarity score to keep
    most_similar_index = np.argmax(similarities)
    most_similar_url = preprocessed_urls[most_similar_index]

    return most_similar_url


def generate_mutamorphic_dataset(x_test, preprocess):
    """Generate a dataset with mutated URLs and their original URLs."""
    mutated_samples = []
    for url in x_test:
        mutated_sample = generate_mutamorphic_sample(url, preprocess)
        mutated_samples.append(mutated_sample)
    return mutated_samples


if __name__ == "__main__":
    with open("dataset/train.txt", "r", encoding="utf-8") as file:
        train = [line.strip() for line in file.readlines()[1:]]
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    from pathlib import Path
    import dvc.api

    params = dvc.api.params_show()

    tokenizer = joblib.load(
        Path(params["dirs"]["outputs"]["preprocess"]) / "tokenizer.joblib"
    )

    def preprocess(x):
        return pad_sequences(tokenizer.texts_to_sequences(x), maxlen=200)

    x_train_mutated = generate_mutamorphic_dataset(raw_x_train, preprocess)

    import os

    mutamorphic_path = Path(params["dirs"]["outputs"]["mutamorphic"])
    os.makedirs(mutamorphic_path, exist_ok=True)
    joblib.dump(x_train_mutated, mutamorphic_path / "x_train.joblib")
