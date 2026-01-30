"""Hotword processing module"""
from src.core.hotword.corrector import PhonemeCorrector, CorrectionResult
from src.core.hotword.phoneme import Phoneme, get_phoneme_info, SIMILAR_PHONEMES

__all__ = ['PhonemeCorrector', 'CorrectionResult', 'Phoneme', 'get_phoneme_info', 'SIMILAR_PHONEMES']
