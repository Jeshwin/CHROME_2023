import tensorflow as tf
from keras import layers
from pylatexenc.latexwalker import (
    LatexWalker,
    LatexMacroNode,
    LatexEnvironmentNode,
    LatexCharsNode,
    LatexGroupNode,
)
import sys
import pickle

START_TOKEN, END_TOKEN = "<START>", "<END>"


# Define the tokenization function using pylatexenc
def latex_tokenizer(latex_string):
    """
    Tokenizes a LaTeX string into tokens using pylatexenc.
    """
    if not latex_string:
        return []
    walker = LatexWalker(latex_string)

    def parse_node(nodelist):
        if len(nodelist) == 0:
            return []
        try:
            tokens = []
            for node in nodelist:
                if not node:
                    continue
                elif node.isNodeType(LatexMacroNode):
                    tokens.append(f"\\{node.macroname}")
                    # Parse arguments if they exist
                    tokens += parse_node(node.nodeargd.argnlist)
                elif node.isNodeType(LatexEnvironmentNode):
                    tokens.append(f"\\begin{{{node.environmentname}}}")
                    tokens += parse_node(node.nodeargd.argnlist)
                    tokens += parse_node(node.nodelist)
                    tokens.append(f"\\end{{{node.environmentname}}}")
                elif node.isNodeType(LatexCharsNode):
                    tokens += list(node.chars)
                elif node.isNodeType(LatexGroupNode):
                    tokens.append(node.delimiters[0])
                    tokens += parse_node(node.nodelist)
                    tokens.append(node.delimiters[1])
            return tokens
        except Exception as e:
            return []

    nodelist, _, _ = walker.get_latex_nodes()
    return parse_node(nodelist)


# Wrap the tokenizer for use in TextVectorization
def tokenize_fn(latex_tensor):
    tokens = []
    for latex_string in latex_tensor:
        tokenized_string = latex_tokenizer(latex_string.numpy().decode("utf-8"))
        tokenized_string.insert(0, START_TOKEN)
        tokenized_string.append(END_TOKEN)
        tokens.append(tokenized_string)
    return tf.ragged.constant(tokens, dtype=tf.string)


# Create a TensorFlow-compatible wrapper
@tf.function
def tf_tokenizer(latex_string):
    return tf.py_function(
        func=tokenize_fn,
        inp=[latex_string],
        Tout=tf.RaggedTensorSpec([None, None], dtype=tf.string),
    )


# Create the TextVectorization layer
max_tokens = 10000  # Adjust depending on your vocabulary size

vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    standardize=None,  # Custom tokenizer, so no built-in preprocessing
    split=tf_tokenizer,
    ragged=True,
)

# TensorFlow implementation

# Adapt the vectorizer on your dataset
dataset = tf.data.TextLineDataset("vocabulary.txt")
dataset = dataset.map(lambda line: [line])
vectorizer.adapt(dataset)

latex_array = [
    r"E = mc^2",
    r"\frac{a}{b} + \sqrt{c}",
    r"\sum_{i=1}^n i^2 = \frac{n(n+1)(2n+1)}{6}",
    r"A = \pi r^2",
    r"G=\begin{bmatrix}1&\dots&1&0&\dots&0\\ \ast&\ast&\ast&&G^{\prime}&\\ \end{bmatrix}",
]
latex_data = tf.constant(latex_array)

# Tokenize and vectorize
tokenized_output = vectorizer(latex_data)
print(tokenized_output)

# De-tokenize
id_to_token = {i: token for i, token in enumerate(vectorizer.get_vocabulary())}


def tokens_to_latex(tokens):
    return "".join([id_to_token[i] for i in tokens])


E_mc2 = tokens_to_latex(tokenized_output(tokenized_output.numpy()[0]))
print(E_mc2)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
