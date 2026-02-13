"""
SEOAnalyzer - Módulo de análisis SEO.
"""
from .analyzer import SEOAnalyzer
from .config import SEOConfig, RecoveryTargets, normalizar_texto
from .comparison import ComparisonAnalyzer
from .recovery import RecoveryAnalyzer
from .metrics import MetricsCalculator
from .utils import (
    filter_by_date_range,
    classify_by_keywords,
    classify_brand,
    classify_kpi,
    compile_pattern,
    parse_date
)

__version__ = '1.0.0'

__all__ = [
    'SEOAnalyzer',
    'SEOConfig',
    'RecoveryTargets',
    'ComparisonAnalyzer',
    'RecoveryAnalyzer',
    'MetricsCalculator',
    'normalizar_texto',
    'filter_by_date_range',
    'classify_by_keywords',
    'classify_brand',
    'classify_kpi',
    'compile_pattern',
    'parse_date'
]
