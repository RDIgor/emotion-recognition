from dataclasses import dataclass


@dataclass
class Face:
    box: list  # absolute
    five_landmarks: tuple  # absolute
    landmarks: tuple  # relative
    applicability: bool
