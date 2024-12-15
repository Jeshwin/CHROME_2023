# Scratch Paper

Paste any useful code from "external sources" here

## LaTeX Token Parser using `pylatexenc`
```py
from pylatexenc.latexwalker import LatexWalker, LatexMacroNode, LatexCharsNode, LatexGroupNode, LatexMathNode

def tokenize_latex(latex_string):
    """
    Tokenizes a LaTeX string into a list of its components (commands, symbols, braces, etc.).
    Args:
    - latex_string (str): The LaTeX string to tokenize.
    
    Returns:
    - List[str]: List of tokens.
    """
    walker = LatexWalker(latex_string)
    nodes, _, _ = walker.get_latex_nodes()

    tokens = []

    def extract_tokens(node):
        if isinstance(node, LatexCharsNode):
            # Split characters into individual symbols
            tokens.extend(list(node.chars))
        elif isinstance(node, LatexMacroNode):
            # Add macro command (e.g., \sqrt)
            tokens.append(f"\\{node.macroname}")
            # Process macro arguments, if any
            if node.nodeoptarg is not None:
                extract_tokens(node.nodeoptarg)
            for arg in node.nodeargs:
                extract_tokens(arg)
        elif isinstance(node, LatexGroupNode):
            # Add opening brace, process content, and add closing brace
            tokens.append("{")
            for child in node.nodelist:
                extract_tokens(child)
            tokens.append("}")
        elif isinstance(node, LatexMathNode):
            # Process math mode content
            tokens.append("$")
            for child in node.nodelist:
                extract_tokens(child)
            tokens.append("$")

    for node in nodes:
        extract_tokens(node)

    return tokens
```
