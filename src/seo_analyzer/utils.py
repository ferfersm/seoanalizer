"""
Utilidades para SEOAnalyzer.
"""
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import polars as pl


def normalizar_texto(texto: str) -> str:
    """Normaliza texto: minúsculas, sin tildes."""
    if not texto:
        return ""
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                   if unicodedata.category(c) != 'Mn')
    return texto.lower().strip()


def compile_pattern(words: List[str]) -> str:
    """Une y escapa palabras en un patrón regex."""
    return '|'.join(map(re.escape, words))


def parse_date(date_str: Union[str, datetime]) -> datetime:
    """Convierte string a datetime."""
    if isinstance(date_str, datetime):
        return date_str
    if isinstance(date_str, str):
        return datetime.strptime(date_str, '%Y-%m-%d')
    raise ValueError(f"Formato de fecha no válido: {date_str}")


def filter_by_date_range(
    df: pl.DataFrame,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    date_col: str = 'date'
) -> pl.DataFrame:
    """Filtra DataFrame por rango de fechas."""
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    col_dtype = df[date_col].dtype
    
    if col_dtype == pl.Utf8:
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        return df.filter(
            (pl.col(date_col) >= start_str) &
            (pl.col(date_col) <= end_str)
        )
    else:
        return df.filter(
            (pl.col(date_col) >= start) &
            (pl.col(date_col) <= end)
        )


def classify_by_keywords(
    df: pl.DataFrame,
    keyword_map: Dict[str, str],
    col: str = 'query'
) -> pl.DataFrame:
    """Clasifica filas por keywords en un diccionario.
    
    Args:
        df: DataFrame a clasificar
        keyword_map: Dict {palabra_normalizada: grupo}
        col: Columna a buscar
    
    Returns:
        DataFrame con columna 'grupo' agregada
    """
    if not keyword_map:
        return df.with_columns([pl.lit(None).alias('grupo')])
    
    pattern = compile_pattern(list(keyword_map.keys()))
    
    df = df.with_columns([
        pl.col(col).str.to_lowercase().alias('_query_lower')
    ])
    
    df = df.with_columns([
        pl.col('_query_lower').str.extract(f'({pattern})', 1).alias('_match')
    ])
    
    def build_expr(matches_dict):
        expr = pl.when(pl.col('_match').is_null()).then(pl.lit(None))
        for palabra, grupo in matches_dict.items():
            expr = expr.when(pl.col('_match') == palabra).then(pl.lit(grupo))
        return expr.otherwise(pl.lit(None))
    
    df = df.with_columns([build_expr(keyword_map).alias('grupo')])
    
    return df.drop('_query_lower', '_match')


def classify_brand(
    df: pl.DataFrame,
    brand_keywords: List[str],
    col: str = 'query'
) -> pl.DataFrame:
    """Clasifica si es brand o non-brand."""
    if not brand_keywords:
        return df.with_columns([pl.lit(False).alias('es_brand')])
    
    brand_norm = [normalizar_texto(kw) for kw in brand_keywords]
    pattern = compile_pattern(brand_norm)
    
    return df.with_columns([
        pl.col(col).str.to_lowercase().str.contains(pattern).alias('es_brand')
    ])


def classify_kpi(
    df: pl.DataFrame,
    kpi_keywords: List[str],
    col: str = 'query'
) -> pl.DataFrame:
    """Clasifica si es keyword importante (KPI)."""
    if not kpi_keywords:
        return df.with_columns([pl.lit(False).alias('es_kpi')])
    
    kpi_norm = [normalizar_texto(kw) for kw in kpi_keywords]
    pattern = compile_pattern(kpi_norm)
    
    return df.with_columns([
        pl.col(col).str.to_lowercase().str.contains(pattern).alias('es_kpi')
    ])


def extract_subdomain(url_col: pl.Series) -> pl.Series:
    """Extrae subdominio de URLs."""
    def get_hostname(url):
        if not url:
            return None
        try:
            if '://' in url:
                host = url.split('://')[1].split('/')[0]
            else:
                host = url.split('/')[0]
            return host.split('.')[0] if '.' in host else host
        except:
            return None
    
    return url_col.map_elements(get_hostname, return_dtype=pl.Utf8)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """División segura que maneja división por cero."""
    return numerator / denominator if denominator != 0 else default


def pct_change(old: float, new: float) -> float:
    """Calcula cambio porcentual."""
    if old == 0:
        return 0.0 if new == 0 else 100.0
    return ((new - old) / old) * 100


def format_number(num: Union[int, float], decimals: int = 0) -> str:
    """Formatea número con separadores de miles."""
    if decimals > 0:
        return f"{num:,.{decimals}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"{num:,}".replace(',', '.')


def safe_sum(df: pl.DataFrame, col: str, default: float = 0.0) -> float:
    """Suma segura de una columna."""
    if df is None or df.is_empty():
        return default
    try:
        val = df.select(pl.col(col).sum()).to_numpy().item()
        return val if val is not None else default
    except:
        return default


def safe_mean(df: pl.DataFrame, col: str, default: float = 0.0) -> float:
    """Promedio seguro de una columna."""
    if df is None or df.is_empty():
        return default
    try:
        val = df.select(pl.col(col).mean()).to_numpy().item()
        return val if val is not None else default
    except:
        return default


def safe_n_unique(df: pl.DataFrame, col: str, default: int = 0) -> int:
    """Cuenta única segura."""
    if df is None or df.is_empty():
        return default
    try:
        val = df.select(pl.col(col).n_unique()).to_numpy().item()
        return val if val is not None else default
    except:
        return default


def safe_value(df: pl.DataFrame, default: float = 0.0) -> float:
    """Obtiene valor único de forma segura."""
    if df is None or df.is_empty():
        return default
    try:
        if df.height == 0:
            return default
        val = df.to_numpy().item()
        return val if val is not None else default
    except:
        return default


def mostrar_df(df: pl.DataFrame, formato: str = 'csv'):
    """
    Muestra un DataFrame según el formato especificado.
    
    Args:
        df: DataFrame de Polars a mostrar
        formato: 'csv' (default) para copiar a Sheets, 'display' para ver en Colab
    
    Ejemplo:
        >>> mostrar_df(df)  # Muestra en formato CSV
        >>> mostrar_df(df, formato='display')  # Muestra como tabla en Colab
    """
    if formato == 'csv':
        print(df.write_csv())
    else:
        try:
            from IPython.display import display
            display(df)
        except ImportError:
            # Si no estamos en IPython/Jupyter, mostrar como CSV
            print(df.write_csv())
