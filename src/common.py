from dataclasses import dataclass
from enum import Enum


@dataclass
class Face:
    box: list  # absolute
    five_landmarks: tuple  # absolute
    landmarks: tuple  # relative
    applicability: bool


class Emotion(Enum):
    SMILE = 0
    NOT_SMILE = 1
