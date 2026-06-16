"""Mapping between EEG motor-imagery classes and wheelchair commands."""

CLASS_TO_COMMAND = {
    "left_hand": "Turn left",
    "right_hand": "Turn right",
    "feet": "Move forward",
    "tongue": "Stop",
}


def classes_to_commands(labels):
    """Translate a sequence of predicted class labels into wheelchair commands."""
    return [CLASS_TO_COMMAND[label] for label in labels]
