"""
SEOAnalyzer - Clase principal.
"""
from typing import Dict, List, Optional, Tuple, Union
import polars as pl
from pathlib import Path

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
        """Preprocesa el DataFrame."""
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
    
    def analyze_queries(self, periodo: str = 'actual') -> pl.DataFrame:
        """Análisis por queries."""
        df = self.get_periodo(periodo)
        
        col_query = self.config.columnas.get('query', 'query')
        
        result = df.group_by(col_query).agg([
            pl.col('clicks').sum().alias('sum_clicks'),
            pl.col('clicks').mean().alias('avg_clicks'),
            pl.col('clicks').median().alias('median_clicks'),
            pl.col('clicks').std().alias('std_clicks'),
            pl.col('impressions').sum().alias('sum_impressions'),
            pl.col('ctr').mean().alias('avg_ctr'),
            pl.col('position').mean().alias('avg_position'),
            pl.col('position').median().alias('median_position'),
            pl.col('position').std().alias('std_position'),
            pl.col('clicks').count().alias('count')
        ])
        
        return result.sort('sum_clicks', descending=True)
    
    def analyze_urls(self, periodo: str = 'actual') -> pl.DataFrame:
        """Análisis por URLs/pages."""
        df = self.get_periodo(periodo)
        
        col_page = self.config.columnas.get('page', 'page')
        
        result = df.group_by(col_page).agg([
            pl.col('clicks').sum().alias('sum_clicks'),
            pl.col('clicks').mean().alias('avg_clicks'),
            pl.col('impressions').sum().alias('sum_impressions'),
            pl.col('ctr').mean().alias('avg_ctr'),
            pl.col('position').mean().alias('avg_position'),
            pl.col('clicks').count().alias('count')
        ])
        
        return result.sort('sum_clicks', descending=True)
    
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
            'ctr': safe_mean(df, 'ctr'),
            'position': safe_mean(df, 'position'),
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
                'ctr': safe_mean(branded, 'ctr'),
                'position': safe_mean(branded, 'position'),
                'queries': safe_n_unique(branded, 'query')
            },
            'non_branded': {
                'clicks': safe_sum(non_branded, 'clicks'),
                'impressions': safe_sum(non_branded, 'impressions'),
                'ctr': safe_mean(non_branded, 'ctr'),
                'position': safe_mean(non_branded, 'position'),
                'queries': safe_n_unique(non_branded, 'query')
            }
        }
    
    def analyze_grupos(self, periodo: str = 'actual') -> Dict:
        """Análisis por grupos de keywords."""
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
        
        for grupo in self.config.grupos.keys():
            df_grupo = df.filter(pl.col('grupo') == grupo)
            
            resultados['grupos'][grupo] = {
                'clicks': safe_sum(df_grupo, 'clicks'),
                'impressions': safe_sum(df_grupo, 'impressions'),
                'ctr': safe_mean(df_grupo, 'ctr'),
                'position': safe_mean(df_grupo, 'position'),
                'queries': safe_n_unique(df_grupo, 'query')
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
            'ctr': safe_mean(kpi_df, 'ctr'),
            'position': safe_mean(kpi_df, 'position'),
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
            pl.col('clicks').sum().alias('clicks_base')
        )
        
        agg_actual = df_actual.group_by(col_query).agg(
            pl.col('clicks').sum().alias('clicks_actual')
        )
        
        merged = agg_base.join(agg_actual, on=col_query, how='full').fill_null(0)
        
        merged = merged.with_columns([
            (pl.col('clicks_actual') - pl.col('clicks_base')).alias('variacion'),
            (pl.col('clicks_actual') - pl.col('clicks_base')).abs().alias('variacion_abs')
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
            for grupo, vals in grupos.items():
                print(f"{grupo}: {vals['clicks']:,} clics, {vals['position']:.1f} posición")
