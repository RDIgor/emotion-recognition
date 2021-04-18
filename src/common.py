from dataclasses import dataclass
from enum import IntEnum


class Emotion(IntEnum):
    NOT_SMILE = 0
    SMILE = 1


@dataclass
class Face:
    box: list  # absolute
    five_landmarks: tuple  # absolute
    landmarks: tuple  # relative
    applicability: bool
    emotion: Emotion

