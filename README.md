# FLoC SimHash

This Python package provides hashing algorithms for computing cohort ids of users based on their browsing history.
As such, it may be used to compute cohort ids of users following Google's **Federated Learning of Cohorts** (FLoC) proposal.

The FLoC proposal is an important part of [The Privacy Sandbox](https://www.chromium.org/Home/chromium-privacy/privacy-sandbox), which is Google's replacement for third-party cookies.
FLoC will enable interest-based advertising, thus preserving an important source of monetization for today's web.

The main idea, as outlined in the [FLoC whitepaper](https://raw.githubusercontent.com/google/ads-privacy/master/proposals/FLoC/FLOC-Whitepaper-Google.pdf), is to replace user cookie ids, which enable user-targeting across multiple sites, by _cohort ids_.
A cohort would consist of a set of users sharing similar browsing behaviour.
By targeting a given cohort, advertisers can ensure that relevant ads are shown while user privacy is preserved by a _hiding in the pack_ mechanism.

The FLoC whitepaper mentions several mechanisms to map users to cohorts, with varying amounts of centralized information.
The algorithms [currently](https://blog.google/products/ads-commerce/2021-01-privacy-sandbox/) being implemented in Google Chrome as a POC are methods based on **SimHash**, which is a type of locality-sensitive hashing initially introduced for detecting near-duplicate documents.

## Contents

<!-- toc -->

- [FLoC SimHash](#floc-simhash)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Individual document-based SimHash](#individual-document-based-simhash)
      - [Providing your own tokenizer](#providing-your-own-tokenizer)
    - [Using the SimHashTransformer in scikit-learn pipelines](#using-the-simhashtransformer-in-scikit-learn-pipelines)
      - [Caveats](#caveats)
  - [Development](#development)

<!-- tocstop -->

## Installation

The `floc-simhash` package is available at PyPI. Install using `pip` as follows.

```bash
pip install floc-simhash
```

The package requires `python>=3.7` and will install `scikit-learn` as a dependency.

## Usage

The package provides two main classes.

- `SimHash`, applying the SimHash algorithm on the md5 hashes of tokens in the given document.

- `SimHashTransformer`, applying the SimHash algorithm to a document vectorization as part of a scikit-learn [pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline)

Finally, there is a third class available:

- `SortingSimHash`, which performs the SortingLSH algorithm by first applying `SimHash` and then clipping the resulting hashes to a given precision.

### Individual document-based SimHash

The `SimHash` class provides a way to calculate the SimHash of any given document, without using any information coming from other documents.

In this case, the document hash is computed by looking at md5 hashes of individual tokens.
We use:

- The implementation of the md5 hashing algorithm available in the `hashlib` module in the [Python standard library](https://docs.python.org/3/library/hashlib.html).

- Bitwise arithmetic for fast computations of the document hash from the individual hashed tokens.

The program below, for example, will print the following hexadecimal string: `cf48b038108e698418650807001800c5`.

```python
from floc_simhash import SimHash

document = "Lorem ipsum dolor sit amet consectetur adipiscing elit"
hashed_document = SimHash(n_bits=128).hash(document)

print(hashed_document)
```

An example more related to computing cohort ids:
the following program computes the cohort id of a user by applying SimHash to the document formed by the pipe-separated list of domains in the user browsing history.

```python
from floc_simhash import SimHash

document = "google.com|hybridtheory.com|youtube.com|reddit.com"
hasher = SimHash(n_bits=128, tokenizer=lambda x: x.split("|"))
hashed_document = hasher.hash(document)

print(hashed_document)
```

The code above will print the hexadecimal string: `14dd1064800880b40025764cd0014715`.

#### Providing your own tokenizer

The `SimHash` constructor will split the given document according to white space by default.
However, it is possible to pass any callable that parses a string into a list of strings in the `tokenizer` parameter.
We have provided an example above where we pass `tokenizer=lambda x: x.split("|")`.

A good example of a more complex tokenization could be passing the [word tokenizer](https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.word_tokenize) in NLTK.
This would be a nice choice if we wished to compute hashes of text documents.

### Using the SimHashTransformer in scikit-learn pipelines

The approach to SimHash outlined in the [FLoC Whitepaper](https://raw.githubusercontent.com/google/ads-privacy/master/proposals/FLoC/FLOC-Whitepaper-Google.pdf) consists of choosing random unit vectors and working on already vectorized data.

The choice of a random unit vector is equivalent to choosing a random hyperplane in feature space.
Choosing `p` random hyperplanes partitions the feature space into `2^p` regions.
Then, a `p`-bit SimHash of a vector encodes the region to which it belongs.

It is reasonable to expect _similar_ documents to have the same hash, provided the vectorization respects the given notion of similarity.

Two vectorizations are discussed in the aforementioned whitepaper: **one-hot** and **tf-idf**; they are available in [scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction).

The `SimHashTransformer` supplies a transformer (implementing the `fit` and `transform` methods) that can be used directly on the output of any of these two vectorizers in order to obtain hashes.

For example, given a 1d-array `X` containing strings, each of them corresponding to a concatenation of the domains visited by a given user and separated by `"|"`, the following code will store in `y` the cohort id of each user, using one-hot encoding and a 32-bit SimHash.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from floc_simhash import SimHashTransformer


X = [
    "google.com|hybridtheory.com|youtube.com|reddit.com",
    "google.com|youtube.com|reddit.com",
    "github.com",
    "google.com|github.com",
]

one_hot_simhash = Pipeline(
    [
        ("vect", CountVectorizer(tokenizer=lambda x: x.split("|"), binary=True)),
        ("simhash", SimHashTransformer(n_bits=32)),
    ]
)

y = one_hot_simhash.fit_transform(X)
```

After running this code, the value of `y` would look similar to the following (expect same lengths; actual hash values depend on the choice of random vectors during `fit`):

```python
['0xd98c7e93' '0xd10b79b3' '0x1085154d' '0x59cd150d']
```

#### Caveats

- The implementation works on the sparse matrices output by `CountVectorizer` and `TfidfTransformer`, in order to manage memory efficiently.

- At the moment, the choice of precision in the numpy arrays results in overflow errors for `p >= 64`. While we are waiting for implementation details of the FLoC POCs, the first indications hint at choices around `p = 50`.

## Development

This project uses [poetry](https://python-poetry.org/) for managing dependencies.

In order to clone the repository and run the unit tests, execute the following steps on an environment with `python>=3.7`.

```bash
git clone https://github.com/hybridtheory/floc-simhash.git
cd floc-simhash
poetry install
pytest
```

The unit tests are property-based, using the [hypothesis](https://hypothesis.readthedocs.io/en/latest/) library.
This allows for algorithm veritication against hundreds or thousands of random generated inputs.

Since running many examples may lengthen the test suite runtime, we also use `pytest-xdist` in order to parallelize the tests.
For example, the following call will run up to 1000 examples for each test with parallelism 4.

```bash
pytest -n 4 --hypothesis-profile=ci
```
