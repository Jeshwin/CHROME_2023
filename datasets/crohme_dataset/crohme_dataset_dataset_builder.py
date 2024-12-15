"""crohme_dataset dataset."""

from pathlib import Path
import xml.etree.ElementTree as ET
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random


def parse_inkml(file_path: Path):
    """
    Parse an InkML file to extract strokes and LaTeX ground truth.

    Args:
        file_path (str): Path to the InkML XML file

    Returns:
        tuple: (strokes, ground_truth)
            - strokes (list of numpy arrays): List of stroke coordinates
              where each stroke is a numpy array of shape (n_points, 2)
            - ground_truth (str): LaTeX ground truth string
    """
    # Parse the XML file
    tree = None
    try:
        tree = ET.parse(file_path)
    except Exception:
        return [], ""
    root = tree.getroot()

    # Namespace handling for InkML
    namespaces = {"ink": "http://www.w3.org/2003/InkML"}

    # Find LaTeX ground truth
    ground_truth = None
    for annotation in root.findall('.//ink:annotation[@type="truth"]', namespaces):
        ground_truth = annotation.text.strip()
        break  # Take the first truth annotation

    # Extract strokes
    strokes = []
    for trace in root.findall(".//ink:trace", namespaces):
        # Split trace points and convert to float coordinates
        points = [
            point.strip().split(" ")[:2] for point in trace.text.strip().split(",")
        ]
        strokes.append(points)

    return strokes, ground_truth


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for crohme_dataset dataset."""

    VERSION = tfds.core.Version("1.0.2")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "1.0.2": "I'm stupid!",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(crohme_dataset): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "strokes": tfds.features.Sequence(
                        tfds.features.Sequence(
                            tfds.features.Tensor(shape=(2,), dtype=tf.float32)
                        )
                    ),
                    "ground_truth": tfds.features.Text(),
                    "filepath": tfds.features.Text(),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("strokes", "ground_truth"),
            homepage="https://github.com/Jeshwin",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(crohme_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "test": self._generate_examples(Path("data/INKML/test")),
            "train": self._generate_examples(Path("data/INKML/train")),
            "validation": self._generate_examples(Path("data/INKML/val")),
        }

    def _generate_examples(self, path: Path):
        """Yields examples."""
        # TODO(crohme_dataset): Yields (key, example) tuples from the dataset
        for f in path.rglob("*.inkml"):
            strokes, ground_truth = parse_inkml(f)
            if len(ground_truth) == 0:
                continue
            yield (
                str(f.resolve()),
                {
                    "strokes": strokes,
                    "ground_truth": ground_truth,
                    "filepath": str(f.resolve()),
                },
            )
