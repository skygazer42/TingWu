"""Hotword processing module"""
from src.core.hotword.corrector import PhonemeCorrector, CorrectionResult
from src.core.hotword.phoneme import Phoneme, get_phoneme_info, SIMILAR_PHONEMES
from src.core.hotword.shape_corrector import ShapeCorrector, JointCorrector

__all__ = [
    'PhonemeCorrector',
    'CorrectionResult',
    'Phoneme',
    'get_phoneme_info',
    'SIMILAR_PHONEMES',
    'ShapeCorrector',
    'JointCorrector',
]
