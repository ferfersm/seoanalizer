"""
SEOAnalyzer - Clase principal.
"""
from typing import Dict, List, Optional, Tuple, Union
import polars as pl
from pathlib import Path
import re

from .config import SEOConfig, RecoveryTargets, normalizar_texto
from .utils import (
    filter_by_date_range,
    classify_by_keywords,
    classify_brand,
    classify_kpi,
    compile_pattern,
    parse_date,
    safe_sum,
    safe_mean,
    safe_n_unique,
    safe_value
)
from .metrics import MetricsCalculator


def _safe_get_value(df: pl.DataFrame, expr, default: float = 0.0) -> float:
    """Obtiene valor de forma segura, maneja DataFrames vacíos."""
    if df.is_empty():
        return default
    try:
        result = df.select(expr)
        if result.is_empty():
            return default
        val = result.to_numpy().item()
        return val if val is not None else default
    except:
        return default


class SEOAnalyzer:
    """Analizador SEO principal.
    
    Uso:
        analyzer = SEOAnalyzer(df)
        analyzer.configure(
            periodo_base=('2025-01-01', '2025-03-31'),
            periodo_actual=('2025-04-01', '2025-06-30'),
            grupos={'Cardiología': ['cardio', 'corazón']},
            brand_keywords=['marca'],
            kpi_keywords=['reserva', 'hora']
        )
        results = analyzer.analyze_all()
    """
    
    def __init__(self, df: pl.DataFrame):
        """Inicializa con un DataFrame de GSC.
        
        Args:
            df: DataFrame con columnas query, page, date, clicks, impressions, ctr, position
        """
        self.df = df
        self.config = SEOConfig()
        self._df_procesado = False
        
        self._preprocess()
    
    @classmethod
    def from_csv(cls, path: str, **kwargs) -> 'SEOAnalyzer':
        """Carga datos desde CSV."""
        df = pl.read_csv(path, **kwargs)
        return cls(df)
    
    @classmethod
    def from_parquet(cls, path: str) -> 'SEOAnalyzer':
        """Carga datos desde Parquet."""
        df = pl.read_parquet(path)
        return cls(df)
    
    def _preprocess(self):
        """Preprocesa el DataFrame y asegura tipos de datos correctos."""
        if self.df.is_empty():
            return
        
        col_fecha = self.config.columnas.get('fecha', 'date')
        
        if col_fecha in self.df.columns:
            if self.df[col_fecha].dtype == pl.Utf8:
                self.df = self.df.with_columns([
                    pl.col(col_fecha).str.to_datetime().alias(col_fecha)
                ])
        
        col_query = self.config.columnas.get('query', 'query')
        if col_query in self.df.columns:
            self.df = self.df.with_columns([
                pl.col(col_query).str.to_lowercase().alias(col_query)
            ])
        
        # Asegurar tipos de datos numéricos correctos
        # clicks e impressions deben ser enteros
        if 'clicks' in self.df.columns:
            self.df = self.df.with_columns([
                pl.col('clicks').cast(pl.Int64).alias('clicks')
            ])
        
        if 'impressions' in self.df.columns:
            self.df = self.df.with_columns([
                pl.col('impressions').cast(pl.Int64).alias('impressions')
            ])
        
        # ctr y position deben ser float
        if 'ctr' in self.df.columns:
            self.df = self.df.with_columns([
                pl.col('ctr').cast(pl.Float64).alias('ctr')
            ])
        
        if 'position' in self.df.columns:
            self.df = self.df.with_columns([
                pl.col('position').cast(pl.Float64).alias('position')
            ])
    
    def configure(
        self,
        periodo_base: Optional[Tuple[str, str]] = None,
        periodo_actual: Optional[Tuple[str, str]] = None,
        grupos: Optional[Dict[str, List[str]]] = None,
        brand_keywords: Optional[List[str]] = None,
        kpi_keywords: Optional[List[str]] = None,
        recovery_targets: Optional[RecoveryTargets] = None,
        cliente: Optional[str] = None
    ):
        """Configura el análisis.
        
        Args:
            periodo_base: (inicio, fin) período base
            periodo_actual: (inicio, fin) período actual
            grupos: Dict {grupo: [keywords]}
            brand_keywords: Keywords de marca
            kpi_keywords: Keywords importantes
            recovery_targets: Targets para recovery
            cliente: Nombre del cliente
        """
        if periodo_base:
            self.config.periodo_base = periodo_base
        if periodo_actual:
            self.config.periodo_actual = periodo_actual
        if grupos:
            self.config.grupos = grupos
        if brand_keywords:
            self.config.brand_keywords = brand_keywords
        if kpi_keywords:
            self.config.kpi_keywords = kpi_keywords
        if recovery_targets:
            self.config.recovery_targets = recovery_targets
        if cliente:
            self.config.cliente = cliente
        
        self._apply_classifications()
        self._df_procesado = True
    
    def _apply_classifications(self):
        """Aplica clasificaciones al DataFrame."""
        if self.df.is_empty():
            self.df = self.df.with_columns([
                pl.lit(None).alias('grupo'),
                pl.lit(False).alias('es_brand'),
                pl.lit(False).alias('es_kpi')
            ])
            return
        
        col_query = self.config.columnas.get('query', 'query')
        
        if self.config.grupos:
            self.df = classify_by_keywords(
                self.df,
                self.config.grupo_map,
                col_query
            )
        else:
            self.df = self.df.with_columns([pl.lit(None).alias('grupo')])
        
        if self.config.brand_keywords:
            self.df = classify_brand(
                self.df,
                self.config.brand_keywords,
                col_query
            )
        else:
            self.df = self.df.with_columns([pl.lit(False).alias('es_brand')])
        
        if self.config.kpi_keywords:
            self.df = classify_kpi(
                self.df,
                self.config.kpi_keywords,
                col_query
            )
        else:
            self.df = self.df.with_columns([pl.lit(False).alias('es_kpi')])
    
    def filter_by_date(
        self,
        start_date: str,
        end_date: str,
        fecha_col: str = None
    ) -> pl.DataFrame:
        """Filtra por rango de fechas."""
        if fecha_col is None:
            fecha_col = self.config.columnas.get('fecha', 'date')
        return filter_by_date_range(self.df, start_date, end_date, fecha_col)
    
    def get_periodo(self, periodo: str) -> pl.DataFrame:
        """Obtiene datos de un período configurado."""
        if periodo == 'base':
            if not self.config.periodo_base:
                raise ValueError("Período base no configurado")
            return self.filter_by_date(*self.config.periodo_base)
        elif periodo == 'actual':
            if not self.config.periodo_actual:
                raise ValueError("Período actual no configurado")
            return self.filter_by_date(*self.config.periodo_actual)
        else:
            raise ValueError(f"Período desconocido: {periodo}")
    
    def analyze_queries(
        self,
        periodo: str = 'actual',
        n: int = 100,
        sort_by: str = 'sum_clicks'
    ) -> pl.DataFrame:
        """Análisis por queries.

        Args:
            periodo: 'base' o 'actual'
            n: Número máximo de queries a retornar (default: 100)
            sort_by: Métrica para ordenar - 'sum_clicks', 'avg_clicks', 'sum_impressions',
                    'avg_ctr', 'avg_position' (default: 'sum_clicks')

        Returns:
            DataFrame con métricas agregadas por query
        """
        valid_metrics = ['sum_clicks', 'avg_clicks', 'median_clicks', 'sum_impressions',
                        'avg_ctr', 'avg_position', 'median_position', 'count']
        if sort_by not in valid_metrics:
            raise ValueError(f"sort_by debe ser uno de: {', '.join(valid_metrics)}")

        df = self.get_periodo(periodo)

        col_query = self.config.columnas.get('query', 'query')

        # Si el DataFrame está vacío, retornar DataFrame con esquema correcto y valores 0
        if df.is_empty():
            return pl.DataFrame({
                col_query: [],
                'sum_clicks': [],
                'avg_clicks': [],
                'median_clicks': [],
                'std_clicks': [],
                'sum_impressions': [],
                'avg_ctr': [],
                'avg_position': [],
                'median_position': [],
                'std_position': [],
                'count': []
            })

        result = df.group_by(col_query).agg([
            pl.col('clicks').sum().cast(pl.Int64).alias('sum_clicks'),
            pl.col('clicks').mean().round(2).alias('avg_clicks'),
            pl.col('clicks').median().round(2).alias('median_clicks'),
            pl.col('clicks').std().round(2).alias('std_clicks'),
            pl.col('impressions').sum().cast(pl.Int64).alias('sum_impressions'),
            # CTR ponderado por impresiones: (ctr * impressions) / sum(impressions)
            ((pl.col('ctr') * pl.col('impressions')).sum() / pl.col('impressions').sum())
            .fill_nan(0.0).round(2).alias('avg_ctr'),
            # Position ponderado por impresiones: (position * impressions) / sum(impressions)
            ((pl.col('position') * pl.col('impressions')).sum() / pl.col('impressions').sum())
            .fill_nan(0.0).round(2).alias('avg_position'),
            pl.col('position').median().round(2).alias('median_position'),
            pl.col('position').std().round(2).alias('std_position'),
            pl.col('clicks').count().cast(pl.Int64).alias('count')
        ])

        if result.is_empty():
            return result

        return result.sort(sort_by, descending=True).head(n)
    
    def get_top10_queries(
        self,
        periodo: str = 'actual',
        n: int = 100,
        sort_by: str = 'clicks'
    ) -> pl.DataFrame:
        """Retorna queries con position <= 10.

        Args:
            periodo: 'base' o 'actual'
            n: Número máximo de queries a retornar (default: 100)
            sort_by: Métrica para ordenar - 'clicks', 'impressions', 'ctr', 'position'
                    (default: 'clicks')

        Returns:
            DataFrame con columns: query, clicks, impressions, ctr, position
        """
        valid_metrics = ['clicks', 'impressions', 'ctr', 'position']
        if sort_by not in valid_metrics:
            raise ValueError(f"sort_by debe ser uno de: {', '.join(valid_metrics)}")

        df = self.get_periodo(periodo)

        col_query = self.config.columnas.get('query', 'query')

        # Si el DataFrame está vacío, retornar DataFrame con esquema correcto y valores 0
        if df.is_empty():
            return pl.DataFrame({
                col_query: [],
                'clicks': [],
                'impressions': [],
                'ctr': [],
                'position': []
            })

        df_filtered = df.filter(pl.col('position') <= 10)

        result = df_filtered.group_by(col_query).agg([
            pl.col('clicks').sum().cast(pl.Int64).alias('clicks'),
            pl.col('impressions').sum().cast(pl.Int64).alias('impressions'),
            # CTR ponderado por impresiones
            ((pl.col('ctr') * pl.col('impressions')).sum() / pl.col('impressions').sum())
            .fill_nan(0.0).round(2).alias('ctr'),
            # Position ponderado por impresiones
            ((pl.col('position') * pl.col('impressions')).sum() / pl.col('impressions').sum())
            .fill_nan(0.0).round(2).alias('position')
        ])

        if result.is_empty():
            return result

        # Para position, ordenar ascendente (menor = mejor)
        descending = sort_by != 'position'
        return result.sort(sort_by, descending=descending).head(n)
    
    def analyze_urls(
        self,
        periodo: str = 'actual',
        n: int = 100,
        sort_by: str = 'sum_clicks'
    ) -> pl.DataFrame:
        """Análisis por URLs/pages.

        Args:
            periodo: 'base' o 'actual'
            n: Número máximo de URLs a retornar (default: 100)
            sort_by: Métrica para ordenar - 'sum_clicks', 'avg_clicks', 'sum_impressions',
                    'avg_ctr', 'avg_position', 'count' (default: 'sum_clicks')

        Returns:
            DataFrame con métricas agregadas por URL
        """
        valid_metrics = ['sum_clicks', 'avg_clicks', 'sum_impressions',
                        'avg_ctr', 'avg_position', 'count']
        if sort_by not in valid_metrics:
            raise ValueError(f"sort_by debe ser uno de: {', '.join(valid_metrics)}")

        df = self.get_periodo(periodo)

        col_page = self.config.columnas.get('page', 'page')

        # Si el DataFrame está vacío, retornar DataFrame con esquema correcto y valores 0
        if df.is_empty():
            return pl.DataFrame({
                col_page: [],
                'sum_clicks': [],
                'avg_clicks': [],
                'sum_impressions': [],
                'avg_ctr': [],
                'avg_position': [],
                'count': []
            })

        result = df.group_by(col_page).agg([
            pl.col('clicks').sum().cast(pl.Int64).alias('sum_clicks'),
            pl.col('clicks').mean().round(2).alias('avg_clicks'),
            pl.col('impressions').sum().cast(pl.Int64).alias('sum_impressions'),
            # CTR ponderado por impresiones
            ((pl.col('ctr') * pl.col('impressions')).sum() / pl.col('impressions').sum())
            .fill_nan(0.0).round(2).alias('avg_ctr'),
            # Position ponderado por impresiones
            ((pl.col('position') * pl.col('impressions')).sum() / pl.col('impressions').sum())
            .fill_nan(0.0).round(2).alias('avg_position'),
            pl.col('clicks').count().cast(pl.Int64).alias('count')
        ])

        if result.is_empty():
            return result

        return result.sort(sort_by, descending=True).head(n)
    
    def analyze_totals(self, periodo: str = 'actual') -> Dict:
        """Análisis de totales."""
        df = self.get_periodo(periodo)
        
        if df.is_empty():
            return {
                'status': 'empty',
                'mensaje': 'No hay datos en el período especificado',
                'clicks': 0,
                'impressions': 0,
                'ctr': 0.0,
                'position': 0.0,
                'queries': 0,
                'pages': 0
            }
        
        return {
            'status': 'ok',
            'clicks': safe_sum(df, 'clicks'),
            'impressions': safe_sum(df, 'impressions'),
            'ctr': round(safe_mean(df, 'ctr'), 2),
            'position': round(safe_mean(df, 'position'), 2),
            'queries': safe_n_unique(df, 'query'),
            'pages': safe_n_unique(df, 'page')
        }
    
    def analyze_brand_vs_nonbrand(self, periodo: str = 'actual') -> Dict:
        """Análisis brand vs non-brand."""
        df = self.get_periodo(periodo)
        
        if df.is_empty():
            return {
                'status': 'empty',
                'mensaje': 'No hay datos en el período especificado',
                'branded': {'clicks': 0, 'impressions': 0, 'ctr': 0.0, 'position': 0.0, 'queries': 0},
                'non_branded': {'clicks': 0, 'impressions': 0, 'ctr': 0.0, 'position': 0.0, 'queries': 0}
            }
        
        branded = df.filter(pl.col('es_brand') == True)
        non_branded = df.filter(pl.col('es_brand') == False)
        
        return {
            'status': 'ok',
            'branded': {
                'clicks': safe_sum(branded, 'clicks'),
                'impressions': safe_sum(branded, 'impressions'),
                'ctr': round(safe_mean(branded, 'ctr'), 2),
                'position': round(safe_mean(branded, 'position'), 2),
                'queries': safe_n_unique(branded, 'query')
            },
            'non_branded': {
                'clicks': safe_sum(non_branded, 'clicks'),
                'impressions': safe_sum(non_branded, 'impressions'),
                'ctr': round(safe_mean(non_branded, 'ctr'), 2),
                'position': round(safe_mean(non_branded, 'position'), 2),
                'queries': safe_n_unique(non_branded, 'query')
            }
        }
    
    def analyze_grupos(self, periodo: str = 'actual') -> Dict:
        """Análisis por grupos de keywords.
        
        Cada query puede pertenecer a múltiples grupos simultáneamente
        si contiene keywords de varios grupos.
        """
        df = self.get_periodo(periodo)
        
        if df.is_empty():
            return {
                'status': 'empty',
                'mensaje': 'No hay datos en el período especificado',
                'grupos': {}
            }
        
        if not self.config.grupos:
            return {'status': 'ok', 'grupos': {}}
        
        resultados = {'status': 'ok', 'grupos': {}}
        col_query = self.config.columnas.get('query', 'query')
        
        # Calcular total de clics para calcular share
        total_clicks = safe_sum(df, 'clicks')
        
        for grupo, keywords in self.config.grupos.items():
            # Compilar patrón para este grupo específico
            pattern = compile_pattern([normalizar_texto(kw) for kw in keywords])
            
            # Filtrar directamente en la query - permite múltiples grupos por query
            df_grupo = df.filter(
                pl.col(col_query).str.to_lowercase().str.contains(pattern)
            )
            
            grupo_clicks = safe_sum(df_grupo, 'clicks')
            share = (grupo_clicks / total_clicks * 100) if total_clicks else 0.0
            
            resultados['grupos'][grupo] = {
                'clicks': grupo_clicks,
                'impressions': safe_sum(df_grupo, 'impressions'),
                'ctr': round(safe_mean(df_grupo, 'ctr'), 2),
                'position': round(safe_mean(df_grupo, 'position'), 2),
                'queries': safe_n_unique(df_grupo, col_query),
                'pages': safe_n_unique(df_grupo, self.config.columnas.get('page', 'page')),
                'share': round(share, 2)
            }
        
        return resultados
    
    def analyze_kpis(self, periodo: str = 'actual') -> Dict:
        """Análisis de keywords KPIs."""
        df = self.get_periodo(periodo)
        
        if df.is_empty():
            return {
                'status': 'empty',
                'mensaje': 'No hay datos en el período especificado',
                'clicks': 0,
                'impressions': 0,
                'ctr': 0.0,
                'position': 0.0,
                'queries': 0
            }
        
        kpi_df = df.filter(pl.col('es_kpi') == True)
        
        return {
            'status': 'ok',
            'clicks': safe_sum(kpi_df, 'clicks'),
            'impressions': safe_sum(kpi_df, 'impressions'),
            'ctr': round(safe_mean(kpi_df, 'ctr'), 2),
            'position': round(safe_mean(kpi_df, 'position'), 2),
            'queries': safe_n_unique(kpi_df, 'query')
        }
    
    def compare_periods(self) -> Dict:
        """Compara período base vs actual."""
        if not self.config.periodo_base or not self.config.periodo_actual:
            raise ValueError("Períodos no configurados. Use configure()")
        
        df_base = self.get_periodo('base')
        df_actual = self.get_periodo('actual')
        
        base_empty = df_base.is_empty()
        actual_empty = df_actual.is_empty()
        
        if base_empty and actual_empty:
            return {
                'status': 'empty',
                'mensaje': 'No hay datos en ningún período',
                'clicks': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0},
                'impressions': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0},
                'ctr': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0},
                'position': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0}
            }
        
        if base_empty:
            return {
                'status': 'error',
                'mensaje': 'Período base sin datos',
                'clicks': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0},
                'impressions': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0},
                'ctr': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0},
                'position': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0}
            }
        
        if actual_empty:
            return {
                'status': 'error',
                'mensaje': 'Período actual sin datos',
                'clicks': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0},
                'impressions': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0},
                'ctr': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0},
                'position': {'base': 0, 'actual': 0, 'var_abs': 0, 'var_pct': 0}
            }
        
        metrics = ['clicks', 'impressions', 'ctr', 'position']
        resultado = {'status': 'ok'}
        
        for metric in metrics:
            if metric in ['clicks', 'impressions']:
                val_base = safe_sum(df_base, metric)
                val_actual = safe_sum(df_actual, metric)
            else:
                val_base = safe_mean(df_base, metric)
                val_actual = safe_mean(df_actual, metric)
            
            var_abs = val_actual - val_base
            var_pct = ((val_actual - val_base) / val_base * 100) if val_base != 0 else 0
            
            resultado[metric] = {
                'base': val_base,
                'actual': val_actual,
                'var_abs': var_abs,
                'var_pct': round(var_pct, 2)
            }
        
        return resultado
    
    def top_queries_variation(self, n: int = 25) -> pl.DataFrame:
        """Top queries con mayor variación."""
        if not self.config.periodo_base or not self.config.periodo_actual:
            raise ValueError("Períodos no configurados")
        
        df_base = self.get_periodo('base')
        df_actual = self.get_periodo('actual')
        
        if df_base.is_empty() or df_actual.is_empty():
            return pl.DataFrame({
                'query': [],
                'clicks_base': [],
                'clicks_actual': [],
                'variacion': [],
                'variacion_abs': []
            })
        
        col_query = self.config.columnas.get('query', 'query')
        
        agg_base = df_base.group_by(col_query).agg(
            pl.col('clicks').sum().cast(pl.Int64).alias('clicks_base')
        )
        
        agg_actual = df_actual.group_by(col_query).agg(
            pl.col('clicks').sum().cast(pl.Int64).alias('clicks_actual')
        )
        
        merged = agg_base.join(agg_actual, on=col_query, how='full').fill_null(0)
        
        merged = merged.with_columns([
            (pl.col('clicks_actual') - pl.col('clicks_base')).cast(pl.Int64).alias('variacion'),
            (pl.col('clicks_actual') - pl.col('clicks_base')).abs().cast(pl.Int64).alias('variacion_abs')
        ])
        
        return merged.sort('variacion_abs', descending=True).head(n)
    
    def analyze_all(self) -> Dict:
        """Ejecuta análisis completo.
        
        Returns:
            Dict con todos los análisis
        """
        results = {
            'configuracion': {
                'cliente': self.config.cliente,
                'periodo_base': self.config.periodo_base,
                'periodo_actual': self.config.periodo_actual
            }
        }
        
        results['totales_actual'] = self.analyze_totals('actual')
        results['totales_base'] = self.analyze_totals('base')
        
        results['comparacion'] = self.compare_periods()
        
        results['brand_vs_nonbrand'] = {
            'actual': self.analyze_brand_vs_nonbrand('actual'),
            'base': self.analyze_brand_vs_nonbrand('base')
        }
        
        if self.config.grupos:
            results['grupos'] = {
                'actual': self.analyze_grupos('actual'),
                'base': self.analyze_grupos('base')
            }
        
        if self.config.kpi_keywords:
            results['kpis'] = {
                'actual': self.analyze_kpis('actual'),
                'base': self.analyze_kpis('base')
            }
        
        results['top_queries_variation'] = self.top_queries_variation()
        
        return results
    
    def print_summary(self):
        """Imprime resumen ejecutivo."""
        if not self._df_procesado:
            print("⚠️  Ejecuta configure() primero")
            return
        
        print("=" * 60)
        print(f"ANÁLISIS SEO - {self.config.cliente}")
        print("=" * 60)
        
        if self.config.periodo_base and self.config.periodo_actual:
            print(f"\nPeríodo Base: {self.config.periodo_base[0]} a {self.config.periodo_base[1]}")
            print(f"Período Actual: {self.config.periodo_actual[0]} a {self.config.periodo_actual[1]}")
        
        print("\n--- TOTALES ---")
        totales = self.analyze_totals('actual')
        print(f"Clics: {totales['clicks']:,}")
        print(f"Impresiones: {totales['impressions']:,}")
        print(f"CTR: {totales['ctr']:.2f}%")
        print(f"Posición: {totales['position']:.1f}")
        
        print("\n--- COMPARACIÓN PERÍODOS ---")
        comp = self.compare_periods()
        for metric, vals in comp.items():
            if metric in ('status', 'mensaje'):
                continue
            print(f"{metric}: {vals['base']:.2f} -> {vals['actual']:.2f} ({vals['var_pct']:+.2f}%)")
        
        if self.config.grupos:
            print("\n--- GRUPOS ---")
            grupos = self.analyze_grupos('actual')
            if 'grupos' in grupos:
                for grupo, vals in grupos['grupos'].items():
                    if isinstance(vals, dict) and 'clicks' in vals:
                        print(f"{grupo}: {vals['clicks']:,} clics, {vals['position']:.1f} posición")

    def _subset_by_keywords(self, df: pl.DataFrame, keywords: List[str]) -> pl.DataFrame:
        """Genera subconjunto filtrando por keywords."""
        if df is None or df.is_empty() or not keywords:
            return df.filter(pl.lit(False))
        
        col_query = self.config.columnas.get('query', 'query')
        pattern = compile_pattern([normalizar_texto(kw) for kw in keywords])
        
        return df.filter(pl.col(col_query).str.contains(pattern))

    def generate_subsets(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """Genera subconjuntos por categorías.
        
        Args:
            df: DataFrame a subdividir
            
        Returns:
            Dict con subconjuntos: 'totales', 'branded', 'non_branded', 
            y subconjuntos por cada grupo configurado
        """
        if df is None or df.is_empty():
            return {
                'totales': df.clone() if df is not None else pl.DataFrame(),
                'branded': pl.DataFrame(),
                'non_branded': pl.DataFrame(),
            }
        
        col_query = self.config.columnas.get('query', 'query')
        
        subsets = {}
        
        subsets['totales'] = df
        
        if self.config.brand_keywords:
            brand_pattern = compile_pattern([normalizar_texto(kw) for kw in self.config.brand_keywords])
            branded = df.filter(pl.col(col_query).str.contains(brand_pattern))
            non_branded = df.filter(~pl.col(col_query).str.contains(brand_pattern))
        else:
            branded = pl.DataFrame()
            non_branded = df
        
        subsets['branded'] = branded
        subsets['non_branded'] = non_branded
        
        if self.config.grupos:
            for grupo, keywords in self.config.grupos.items():
                # Filtrar directamente por keywords - permite múltiples grupos por query
                pattern = compile_pattern([normalizar_texto(kw) for kw in keywords])
                df_grupo = df.filter(pl.col(col_query).str.to_lowercase().str.contains(pattern))
                subsets[f'solo {grupo}'] = df_grupo
        
        if self.config.kpi_keywords:
            kpi_pattern = compile_pattern([normalizar_texto(kw) for kw in self.config.kpi_keywords])
            subsets['kpi_keywords'] = df.filter(pl.col(col_query).str.contains(kpi_pattern))
        
        return subsets

    def summarize_dataframes(self, dfs: Dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Resume DataFrames con métricas.
        
        Args:
            dfs: Dict de {nombre: DataFrame}
            
        Returns:
            DataFrame con columnas: data, clicks, impressions, ctr, 
            posicion, share, posicion_mediana
        """
        if not dfs or 'totales' not in dfs:
            return pl.DataFrame({
                'data': [],
                'clics': [],
                'impresiones': [],
                'ctr': [],
                'posicion': [],
                'share': [],
                'posicion_mediana': []
            })
        
        total_clicks = safe_sum(dfs.get('totales'), 'clicks', 0)
        
        rows = []
        for name, d in dfs.items():
            if d is None or d.is_empty():
                rows.append({
                    'data': name,
                    'clics': 0,
                    'impresiones': 0,
                    'ctr': 0.0,
                    'posicion': 0.0,
                    'share': 0.0,
                    'posicion_mediana': 0.0
                })
                continue
            
            clicks = safe_sum(d, 'clicks')
            impressions = safe_sum(d, 'impressions')
            
            ctr_w = self._weighted_avg_ctr(d) if impressions else 0.0
            pos_w = self._weighted_avg_position(d) if impressions else 0.0
            pos_median = safe_mean(d, 'position', 0.0)
            
            rows.append({
                'data': name,
                'clics': clicks,
                'impresiones': impressions,
                'ctr': round(ctr_w, 2),
                'posicion': round(pos_w, 1),
                'share': round((clicks / total_clicks * 100) if total_clicks else 0, 2),
                'posicion_mediana': round(pos_median, 1)
            })
        
        return pl.DataFrame(rows)

    def _weighted_avg_ctr(self, df: pl.DataFrame) -> float:
        """Calcula CTR promedio ponderado por impresiones."""
        if df is None or df.is_empty():
            return 0.0
        try:
            impressions = df['impressions'].to_numpy()
            ctr = df['ctr'].to_numpy()
            total_imp = impressions.sum()
            if total_imp == 0:
                return 0.0
            return (ctr * impressions).sum() / total_imp
        except:
            return safe_mean(df, 'ctr', 0.0)

    def _weighted_avg_position(self, df: pl.DataFrame) -> float:
        """Calcula posición promedio ponderada por impresiones."""
        if df is None or df.is_empty():
            return 0.0
        try:
            impressions = df['impressions'].to_numpy()
            position = df['position'].to_numpy()
            total_imp = impressions.sum()
            if total_imp == 0:
                return 0.0
            return (position * impressions).sum() / total_imp
        except:
            return safe_mean(df, 'position', 0.0)

    def compare_summaries(
        self,
        df_inicio: pl.DataFrame,
        df_fin: pl.DataFrame,
        metrics: Tuple[str, ...] = ('clics', 'impresiones', 'ctr', 'posicion', 'share')
    ) -> pl.DataFrame:
        """Compara resúmenes entre períodos.
        
        Args:
            df_inicio: DataFrame de resumen del período inicio
            df_fin: DataFrame de resumen del período fin
            metrics: Métricas a comparar
            
        Returns:
            DataFrame con variaciones absolutas y porcentuales
        """
        if df_inicio is None or df_inicio.is_empty():
            df_inicio = pl.DataFrame({'data': [], 'clics': [], 'impresiones': []})
        if df_fin is None or df_fin.is_empty():
            df_fin = pl.DataFrame({'data': [], 'clics': [], 'impresiones': []})
        
        col_data = 'data'
        
        cmp = df_inicio.join(df_fin, on=col_data, how='full', suffix='_fin').fill_null(0)
        
        metric_cols = []
        for m in metrics:
            col_ini = m if m in cmp.columns else f'{m}_ini'
            col_fin = f'{m}_fin' if f'{m}_fin' in cmp.columns else m
            
            if col_ini in cmp.columns and col_fin in cmp.columns:
                cmp = cmp.with_columns([
                    (pl.col(col_fin) - pl.col(col_ini)).alias(f'Variacion_{m}')
                ])
                
                cmp = cmp.with_columns([
                    pl.when(pl.col(col_ini) != 0)
                    .then((pl.col(f'Variacion_{m}') / pl.col(col_ini)) * 100)
                    .otherwise(0.0)
                    .alias(f'Variacion_porcentual_{m}')
                ])
                
                metric_cols.extend([col_ini, col_fin, f'Variacion_{m}', f'Variacion_porcentual_{m}'])
        
        rename_map = {}
        for m in metrics:
            col_ini = m if m in df_inicio.columns else m
            col_fin = f'{m}_fin' if f'{m}_fin' in cmp.columns else m
            if col_ini in cmp.columns:
                rename_map[col_ini] = f'{m} INI'
            if col_fin in cmp.columns:
                rename_map[col_fin] = f'{m} FIN'
        
        if rename_map:
            cmp = cmp.rename(rename_map)
        
        order_cols = [col_data]
        for m in metrics:
            for suffix in ('INI', 'FIN', ''):
                col_name = f'{m} {suffix}' if suffix else m
                if col_name in cmp.columns:
                    order_cols.append(col_name)
        
        existing_cols = [c for c in order_cols if c in cmp.columns]
        return cmp.select(existing_cols)

    def create_comparison_df(
        self,
        df_curr: pl.DataFrame,
        df_prev: pl.DataFrame,
        labels: List[List[str]],
        metric: str = 'clicks'
    ) -> pl.DataFrame:
        """Crea DataFrame comparativo por marca/grupo.
        
        Args:
            df_curr: DataFrame período actual
            df_prev: DataFrame período previo
            labels: Lista de listas de keywords por grupo
            metric: Métrica a comparar ('clicks' o 'impressions')
            
        Returns:
            DataFrame comparativo con variaciones
        """
        if df_curr is None:
            df_curr = pl.DataFrame()
        if df_prev is None:
            df_prev = pl.DataFrame()
        
        col_query = self.config.columnas.get('query', 'query')
        
        metric_curr = metric if metric in df_curr.columns else 'clicks'
        metric_prev = metric if metric in df_prev.columns else 'clicks'
        
        tot_curr = safe_sum(df_curr, metric_curr, 0)
        tot_prev = safe_sum(df_prev, metric_prev, 0)
        
        data = []
        for label in labels:
            if not label:
                continue
                
            pattern = compile_pattern([normalizar_texto(kw) for kw in label])
            
            curr_val = 0
            prev_val = 0
            
            if not df_curr.is_empty():
                filtered_curr = df_curr.filter(pl.col(col_query).str.contains(pattern))
                curr_val = safe_sum(filtered_curr, metric_curr, 0)
            
            if not df_prev.is_empty():
                filtered_prev = df_prev.filter(pl.col(col_query).str.contains(pattern))
                prev_val = safe_sum(filtered_prev, metric_prev, 0)
            
            var = curr_val - prev_val
            var_pct = round((var / prev_val * 100), 2) if prev_val != 0 else 0.0
            share_curr = round((curr_val / tot_curr * 100), 2) if tot_curr else 0.0
            share_prev = round((prev_val / tot_prev * 100), 2) if tot_prev else 0.0
            
            data.append({
                'Marca': ', '.join(label[:3]) + ('...' if len(label) > 3 else ''),
                f'{metric.capitalize()} Periodo Actual': curr_val,
                f'{metric.capitalize()} Periodo Previo': prev_val,
                f'Variación {metric.capitalize()}': var,
                f'Variación Porcentual {metric.capitalize()} (%)': var_pct,
                f'Share {metric.capitalize()} Periodo Actual (%)': share_curr,
                f'Share {metric.capitalize()} Periodo Previo (%)': share_prev,
                f'Variación Share {metric.capitalize()}': round(share_curr - share_prev, 2)
            })
        
        return pl.DataFrame(data)

    def analyze_subdomains(
        self,
        df_prev: pl.DataFrame,
        df_curr: pl.DataFrame,
        subdomain_patterns: List[str]
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Análisis por subdominios.
        
        Args:
            df_prev: DataFrame período previo
            df_curr: DataFrame período actual
            subdomain_patterns: Lista de patrones de subdominios
            
        Returns:
            Tuple de (DataFrame prev, DataFrame curr) con métricas por subdominio
        """
        if df_prev is None:
            df_prev = pl.DataFrame()
        if df_curr is None:
            df_curr = pl.DataFrame()
        
        col_page = self.config.columnas.get('page', 'page')
        
        def get_subdomain_summary(df, patterns):
            if df is None or df.is_empty():
                return pl.DataFrame({'subdomain': [], 'clicks': [], 'impressions': []})
            
            rows = []
            for pattern in patterns:
                filtered = df.filter(pl.col(col_page).str.contains(pattern))
                if not filtered.is_empty():
                    rows.append({
                        'subdomain': pattern,
                        'clicks': safe_sum(filtered, 'clicks'),
                        'impressions': safe_sum(filtered, 'impressions')
                    })
            
            if not rows:
                return pl.DataFrame({'subdomain': [], 'clicks': [], 'impressions': []})
            
            return pl.DataFrame(rows)
        
        prev_summary = get_subdomain_summary(df_prev, subdomain_patterns)
        curr_summary = get_subdomain_summary(df_curr, subdomain_patterns)
        
        return prev_summary, curr_summary

    def top_n_queries_by_variation(
        self,
        df_prev: pl.DataFrame,
        df_curr: pl.DataFrame,
        n: int = 10,
        metric: str = 'clicks',
        includes: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        exact_match: bool = False,
        subdomain: Optional[str] = None
    ) -> pl.DataFrame:
        """Top queries con mayor variación entre períodos.
        
        Args:
            df_prev: DataFrame período previo
            df_curr: DataFrame período actual
            n: Número de resultados
            metric: Métrica ('clicks' o 'impressions')
            includes: Filtrar solo queries que contengan estos términos
            excludes: Excluir queries que contengan estos términos
            exact_match: Usar match exacto para includes/excludes
            subdomain: Filtrar por subdominio
            
        Returns:
            DataFrame con query, grupo, metric_prev, metric_curr, diferencia, abs_diff
        """
        if metric not in ['clicks', 'impressions']:
            raise ValueError("metric debe ser 'clicks' o 'impressions'")
        
        if df_prev is None:
            df_prev = pl.DataFrame()
        if df_curr is None:
            df_curr = pl.DataFrame()
        
        col_query = self.config.columnas.get('query', 'query')
        col_page = self.config.columnas.get('page', 'page')
        
        d1 = df_prev.clone()
        d2 = df_curr.clone()
        
        if subdomain:
            pattern = compile_pattern([subdomain]) if isinstance(subdomain, (list, tuple)) else re.escape(subdomain)
            d1 = d1.filter(pl.col(col_page).str.contains(pattern))
            d2 = d2.filter(pl.col(col_page).str.contains(pattern))
        
        def apply_filter(df, terms, include=True):
            if not terms:
                return df
            if exact_match:
                terms_lower = [t.lower() for t in (terms if isinstance(terms, list) else [terms])]
                return df.filter(pl.col(col_query).str.to_lowercase().is_in(terms_lower))
            else:
                pattern = compile_pattern(terms if isinstance(terms, list) else [terms])
                mask = df[col_query].str.contains(pattern)
                return df.filter(mask if include else ~mask)
        
        if includes:
            d1 = apply_filter(d1, includes, True)
            d2 = apply_filter(d2, includes, True)
        
        if excludes:
            d1 = apply_filter(d1, excludes, False)
            d2 = apply_filter(d2, excludes, False)
        
        brand_col = 'grupo' if 'grupo' in d1.columns else 'matched_brand'
        
        a1 = d1.group_by([col_query, brand_col]).agg(
            pl.col(metric).sum().alias(f'{metric}_prev')
        ) if not d1.is_empty() else pl.DataFrame({
            col_query: [], brand_col: [], f'{metric}_prev': []
        }).with_columns([
            pl.col(f'{metric}_prev').cast(pl.Int64)
        ])
        
        a2 = d2.group_by([col_query, brand_col]).agg(
            pl.col(metric).sum().alias(f'{metric}_curr')
        ) if not d2.is_empty() else pl.DataFrame({
            col_query: [], brand_col: [], f'{metric}_curr': []
        }).with_columns([
            pl.col(f'{metric}_curr').cast(pl.Int64)
        ])
        
        if a1.is_empty() and a2.is_empty():
            return pl.DataFrame({
                'query': [],
                'grupo': [],
                f'{metric}_prev': [],
                f'{metric}_curr': [],
                'diferencia': [],
                'abs_diff': []
            })
        
        merged = a1.join(a2, on=[col_query, brand_col], how='full')
        merged = merged.with_columns([
            pl.col(f'{metric}_prev').fill_null(0),
            pl.col(f'{metric}_curr').fill_null(0),
            pl.col(brand_col).fill_null("sin_categoria")
        ])
        
        merged = merged.with_columns([
            (pl.col(f'{metric}_curr') - pl.col(f'{metric}_prev')).alias('diferencia'),
            (pl.col(f'{metric}_curr') - pl.col(f'{metric}_prev')).abs().alias('abs_diff')
        ])
        
        result = merged.sort('abs_diff', descending=True).head(n)
        
        return result.rename({brand_col: 'grupo'}).select([
            col_query, 'grupo', f'{metric}_prev', f'{metric}_curr', 'diferencia', 'abs_diff'
        ])

    def traffic_distribution_by_keyword_category(self, df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Distribución del tráfico por categoría de keyword.
        
        Args:
            df: DataFrame opcional. Si es None, usa el DataFrame completo.
            
        Returns:
            DataFrame con keyword_category, clicks, impressions
        """
        if df is None:
            df = self.df
        
        if df is None or df.is_empty():
            return pl.DataFrame({
                'keyword_category': [],
                'clicks': [],
                'impressions': []
            })
        
        if 'grupo' not in df.columns:
            df = df.with_columns([pl.lit('no_grupo').alias('grupo')])
        
        if 'es_brand' not in df.columns:
            df = df.with_columns([pl.lit(False).alias('es_brand')])
        
        summary = df.group_by('grupo').agg(
            pl.col('clicks').sum().alias('clicks'),
            pl.col('impressions').sum().alias('impressions')
        ).rename({'grupo': 'keyword_category'}).with_columns(
            pl.col('keyword_category').fill_null("sin_categoria")
        )
        
        brand_clicks = safe_sum(df.filter(pl.col('es_brand') == True), 'clicks', 0)
        brand_impr = safe_sum(df.filter(pl.col('es_brand') == True), 'impressions', 0)
        
        if brand_clicks > 0:
            brand_row = pl.DataFrame({
                'keyword_category': ['branded'],
                'clicks': [brand_clicks],
                'impressions': [brand_impr]
            })
            summary = pl.concat([summary, brand_row])
        
        return summary.sort('clicks', descending=True)

    def traffic_distribution_by_subdomain(self, df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Distribución del tráfico por subdominio.
        
        Args:
            df: DataFrame opcional. Si es None, usa el DataFrame completo.
            
        Returns:
            DataFrame con subdomain, clicks, impressions
        """
        if df is None:
            df = self.df
        
        if df is None or df.is_empty():
            return pl.DataFrame({
                'subdomain': [],
                'clicks': [],
                'impressions': []
            })
        
        col_page = self.config.columnas.get('page', 'page')
        
        df_with_subdomain = df.with_columns([
            self._extract_hostname(pl.col(col_page)).alias('subdomain')
        ]).filter(pl.col('subdomain').is_not_null())
        
        if df_with_subdomain.is_empty():
            return pl.DataFrame({
                'subdomain': [],
                'clicks': [],
                'impressions': []
            })
        
        summary = df_with_subdomain.group_by('subdomain').agg(
            pl.col('clicks').sum().alias('clicks'),
            pl.col('impressions').sum().alias('impressions')
        ).sort('clicks', descending=True)
        
        return summary

    def _extract_hostname(self, url_series) -> pl.Series:
        """Extrae hostname de URLs."""
        def get_hostname(url):
            if not url:
                return None
            try:
                if '://' in url:
                    host = url.split('://')[1].split('/')[0]
                else:
                    host = url.split('/')[0]
                return host
            except:
                return None
        
        return url_series.map_elements(get_hostname, return_dtype=pl.Utf8)

    def resumen_kw(
        self,
        kw: str | None = None,
        col: str = 'query',
        periodo: str = 'month',
        rango: tuple = (None, None),
        exact_match: bool = True
    ) -> pl.DataFrame:
        """
        Resumen de métricas para una keyword (o para todo el df si kw=None).
        
        Args:
            kw: Keyword a filtrar. Si None, usa todo el DataFrame.
            col: Columna para filtrar ('query' o 'page')
            periodo: 'month' o 'day' para agrupación temporal
            rango: Tupla (fecha_inicio, fecha_fin) para completar rango
            exact_match: True para match exacto, False para contains
            
        Returns:
            DataFrame con métricas por período y variaciones
        """
        from typing import Literal
        
        if periodo not in ['month', 'day']:
            raise ValueError("periodo debe ser 'month' o 'day'")
        
        # Obtener período actual (o usar todo el df si no hay período configurado)
        if hasattr(self.config, 'periodo_actual') and self.config.periodo_actual:
            df = self.get_periodo('actual')
        else:
            df = self.df
        
        # Si el DataFrame está vacío, retornar DataFrame vacío con esquema correcto
        if df.is_empty():
            return pl.DataFrame({
                'period': [],
                'clicks': [],
                'impressions': [],
                'ctr': [],
                'avg_position': [],
                'variacion_clicks': [],
                'variacion_impressions': [],
                'variacion_clicks_pct': [],
                'variacion_impressions_pct': []
            })
        
        # Filtrar por keyword si se especifica
        if kw is not None:
            col_name = self.config.columnas.get(col, col) if hasattr(self.config, 'columnas') else col
            if exact_match:
                df = df.filter(pl.col(col_name) == kw)
            else:
                df = df.filter(pl.col(col_name).str.contains(kw))
        
        # Crear columna de período
        if periodo == 'month':
            df = df.with_columns([
                pl.col('date').dt.truncate('1mo').alias('period')
            ])
        else:  # day
            df = df.with_columns([
                pl.col('date').dt.truncate('1d').alias('period')
            ])
        
        # Agrupar y calcular métricas
        result = df.group_by('period').agg([
            pl.col('clicks').sum().alias('clicks'),
            pl.col('impressions').sum().alias('impressions'),
            pl.col('position').mean().round(2).alias('avg_position')
        ]).with_columns([
            # CTR calculado correctamente: clicks / impressions
            (pl.col('clicks') / pl.col('impressions')).fill_nan(0.0).round(2).alias('ctr')
        ]).sort('period')
        
        # Completar rango si se especifica
        start, end = rango
        if (start or end) and not result.is_empty():
            # Convertir a string para date_range
            start_date = start if start else result['period'].min()
            end_date = end if end else result['period'].max()
            
            # Crear rango completo de fechas
            freq = '1mo' if periodo == 'month' else '1d'
            all_periods = pl.date_range(
                start=pl.lit(start_date).str.to_datetime(),
                end=pl.lit(end_date).str.to_datetime(),
                interval=freq,
                eager=True
            )
            
            # Join con rango completo para rellenar vacíos con 0
            periods_df = pl.DataFrame({'period': all_periods})
            result = periods_df.join(result, on='period', how='left').fill_null(0)
            
            # Recalcular CTR después de fill_null
            result = result.with_columns([
                (pl.col('clicks') / pl.col('impressions')).fill_nan(0.0).round(2).alias('ctr')
            ])
        
        # Calcular variaciones
        result = result.with_columns([
            pl.col('clicks').diff().fill_null(0).cast(pl.Int64).alias('variacion_clicks'),
            pl.col('impressions').diff().fill_null(0).cast(pl.Int64).alias('variacion_impressions'),
            pl.col('clicks').pct_change().fill_null(0).round(2).alias('variacion_clicks_pct'),
            pl.col('impressions').pct_change().fill_null(0).round(2).alias('variacion_impressions_pct')
        ])
        
        return result
