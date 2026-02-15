"""
RecoveryAnalyzer - Análisis de recovery post-migración.
"""
from typing import Dict, List, Optional, Tuple
import polars as pl

from .config import SEOConfig, RecoveryTargets
from .utils import filter_by_date_range, compile_pattern, normalizar_texto, classify_by_keywords, classify_brand, classify_kpi, safe_sum, safe_n_unique


class RecoveryAnalyzer:
    """Analizador de recovery SEO post-migración.
    
    Uso:
        recovery = RecoveryAnalyzer(df, config, targets)
        recovery.apply_classifications()  # Si no están aplicadas
        recovery.analyze()
    """
    
    def __init__(
        self,
        df: pl.DataFrame,
        config: SEOConfig,
        targets: RecoveryTargets
    ):
        """Inicializa con DataFrame, config y targets."""
        self.df = df
        self.config = config
        self.targets = targets
        self._classifications_applied = 'es_brand' in df.columns
    
    def apply_classifications(self):
        """Aplica clasificaciones al DataFrame si no están aplicadas."""
        if self._classifications_applied:
            return
        
        if self.df.is_empty():
            self.df = self.df.with_columns([
                pl.lit(None).alias('grupo'),
                pl.lit(False).alias('es_brand'),
                pl.lit(False).alias('es_kpi')
            ])
            self._classifications_applied = True
            return
        
        col_query = self.config.columnas.get('query', 'query')
        
        if self.config.grupos:
            self.df = classify_by_keywords(self.df, self.config.grupo_map, col_query)
        else:
            self.df = self.df.with_columns([pl.lit(None).alias('grupo')])
        
        if self.config.brand_keywords:
            self.df = classify_brand(self.df, self.config.brand_keywords, col_query)
        else:
            self.df = self.df.with_columns([pl.lit(False).alias('es_brand')])
        
        if self.config.kpi_keywords:
            self.df = classify_kpi(self.df, self.config.kpi_keywords, col_query)
        else:
            self.df = self.df.with_columns([pl.lit(False).alias('es_kpi')])
        
        self._classifications_applied = True
    
    def calculate_traffic_recovery(self) -> Dict:
        """Calcula recuperación de tráfico."""
        if not self.config.periodo_base or not self.config.periodo_actual:
            raise ValueError("Períodos no configurados")
        
        if not self.targets.target_trafico:
            return {'status': 'no_target'}
        
        df_base = filter_by_date_range(
            self.df,
            self.config.periodo_base[0],
            self.config.periodo_base[1]
        )
        
        df_actual = filter_by_date_range(
            self.df,
            self.config.periodo_actual[0],
            self.config.periodo_actual[1]
        )
        
        clics_base = df_base.select(pl.col('clicks').sum()).to_numpy().item()
        clics_actual = df_actual.select(pl.col('clicks').sum()).to_numpy().item()
        
        target = self.targets.target_trafico
        progress_pct = (clics_actual / target * 100) if target > 0 else 0
        
        recovery_pct = (clics_actual / clics_base * 100) if clics_base > 0 else 0
        
        return {
            'base': clics_base,
            'actual': clics_actual,
            'target': target,
            'progress_pct': round(progress_pct, 2),
            'recovery_pct': round(recovery_pct, 2),
            'status': 'on_track' if progress_pct >= 70 else 'at_risk' if progress_pct >= 50 else 'critical'
        }
    
    def calculate_transactional_impressions(self) -> Dict:
        """Calcula impresiones de keywords transaccionales."""
        if not self.config.periodo_base or not self.config.periodo_actual:
            raise ValueError("Períodos no configurados")
        
        if not self.targets.keywords_transaccionales:
            return {'status': 'no_keywords'}
        
        df_base = filter_by_date_range(
            self.df,
            self.config.periodo_base[0],
            self.config.periodo_base[1]
        )
        
        df_actual = filter_by_date_range(
            self.df,
            self.config.periodo_actual[0],
            self.config.periodo_actual[1]
        )
        
        kws = [normalizar_texto(k) for k in self.targets.keywords_transaccionales]
        pattern = compile_pattern(kws)
        
        df_base_trans = df_base.filter(
            pl.col('query').str.contains(pattern)
        )
        df_actual_trans = df_actual.filter(
            pl.col('query').str.contains(pattern)
        )
        
        imp_base = df_base_trans.select(pl.col('impressions').sum()).to_numpy().item()
        imp_actual = df_actual_trans.select(pl.col('impressions').sum()).to_numpy().item()
        
        var_pct = ((imp_actual - imp_base) / imp_base * 100) if imp_base > 0 else 0
        
        return {
            'base': imp_base,
            'actual': imp_actual,
            'variation': {
                'abs': imp_actual - imp_base,
                'pct': round(var_pct, 2)
            }
        }
    
    def calculate_nonbranded_coverage(self) -> Dict:
        """Calcula cobertura non-branded."""
        if not self.config.periodo_base or not self.config.periodo_actual:
            raise ValueError("Períodos no configurados")
        
        df_base = filter_by_date_range(
            self.df,
            self.config.periodo_base[0],
            self.config.periodo_base[1]
        )
        
        df_actual = filter_by_date_range(
            self.df,
            self.config.periodo_actual[0],
            self.config.periodo_actual[1]
        )
        
        df_base_nb = df_base.filter(pl.col('es_brand') == False)
        df_actual_nb = df_actual.filter(pl.col('es_brand') == False)
        
        clics_base = df_base_nb.select(pl.col('clicks').sum()).to_numpy().item()
        clics_actual = df_actual_nb.select(pl.col('clicks').sum()).to_numpy().item()
        
        queries_base = df_base_nb.select(pl.col('query').n_unique()).to_numpy().item()
        queries_actual = df_actual_nb.select(pl.col('query').n_unique()).to_numpy().item()
        
        recovery_pct = (clics_actual / clics_base * 100) if clics_base > 0 else 0
        
        return {
            'base': {
                'clicks': clics_base,
                'queries': queries_base
            },
            'actual': {
                'clicks': clics_actual,
                'queries': queries_actual
            },
            'recovery_pct': round(recovery_pct, 2),
            'status': 'recovered' if recovery_pct >= 100 else 'recovering' if recovery_pct >= 70 else 'critical'
        }
    
    def calculate_url_optimization(self) -> Dict:
        """Calcula progreso de optimización de URLs."""
        if not self.targets.total_urls_sitio:
            return {'status': 'no_target'}
        
        total = self.targets.total_urls_sitio
        optimizadas = len(self.targets.urls_ya_optimizadas)
        
        progress_pct = (optimizadas / total * 100) if total > 0 else 0
        
        return {
            'total_urls': total,
            'urls_optimizadas': optimizadas,
            'progress_pct': round(progress_pct, 2),
            'status': 'on_track' if progress_pct >= self.targets.target_optimizacion_urls else 'in_progress'
        }
    
    def calculate_top10_coverage(self) -> Dict:
        """Calcula % de keywords en top 10.
        
        Returns:
            Dict con total_queries, top10_queries, coverage_pct, target, status,
            y top10_df (DataFrame con queries y URLs en top 10)
        """
        if not self.config.periodo_actual:
            raise ValueError("Período actual no configurado")
        
        df_actual = filter_by_date_range(
            self.df,
            self.config.periodo_actual[0],
            self.config.periodo_actual[1]
        )
        
        if df_actual.is_empty():
            return {
                'total_queries': 0,
                'top10_queries': 0,
                'coverage_pct': 0.0,
                'target': self.targets.target_top10_coverage,
                'status': 'no_data',
                'top10_df': pl.DataFrame({
                    'query': [],
                    'page': [],
                    'clicks': [],
                    'impressions': [],
                    'position': []
                })
            }
        
        total_queries = df_actual.select(pl.col('query').n_unique()).to_numpy().item()
        
        top10 = df_actual.filter(pl.col('position') <= 10)
        top10_queries = top10.select(pl.col('query').n_unique()).to_numpy().item()
        
        coverage_pct = (top10_queries / total_queries * 100) if total_queries > 0 else 0
        
        top10_df = top10.group_by(['query', 'page']).agg([
            pl.col('clicks').sum().alias('clicks'),
            pl.col('impressions').sum().alias('impressions'),
            pl.col('position').mean().alias('position')
        ]).sort('clicks', descending=True)
        
        return {
            'total_queries': total_queries,
            'top10_queries': top10_queries,
            'coverage_pct': round(coverage_pct, 2),
            'target': self.targets.target_top10_coverage,
            'status': 'ok' if coverage_pct >= self.targets.target_top10_coverage else 'below_target',
            'top10_df': top10_df
        }
    
    def detect_cannibalization(
        self,
        max_queries: int = 100,
        max_urls_per_query: int = 10,
        sort_by: str = 'impressions'
    ) -> Dict:
        """Detecta canibalización de keywords.
        
        Args:
            max_queries: Número máximo de queries canibalizadas a retornar (default: 100)
            max_urls_per_query: Número máximo de URLs a mostrar por query (default: 10)
            sort_by: Criterio de ordenamiento - 'impressions' (default) o 'clicks'
        
        Returns:
            Dict con cannibalized_queries, total_queries, cannibalization_rate,
            target, status, y cannibal_df (DataFrame con queries canibalizadas y URLs)
        """
        if sort_by not in ['impressions', 'clicks']:
            raise ValueError("sort_by debe ser 'impressions' o 'clicks'")
        
        if not self.config.periodo_actual:
            raise ValueError("Período actual no configurado")
        
        df_actual = filter_by_date_range(
            self.df,
            self.config.periodo_actual[0],
            self.config.periodo_actual[1]
        )
        
        if df_actual.is_empty():
            return {
                'cannibalized_queries': 0,
                'total_queries': 0,
                'cannibalization_rate': 0.0,
                'target': self.targets.target_cannibalization,
                'status': 'no_data',
                'cannibal_df': pl.DataFrame({
                    'query': [],
                    'url_count': [],
                    'total_clicks': [],
                    'total_impressions': [],
                    'urls': []
                })
            }
        
        top20 = df_actual.filter(pl.col('position') <= 20)
        
        if top20.is_empty():
            return {
                'cannibalized_queries': 0,
                'total_queries': df_actual.select(pl.col('query').n_unique()).to_numpy().item(),
                'cannibalization_rate': 0.0,
                'target': self.targets.target_cannibalization,
                'status': 'no_data',
                'cannibal_df': pl.DataFrame({
                    'query': [],
                    'url_count': [],
                    'total_clicks': [],
                    'total_impressions': [],
                    'urls': []
                })
            }
        
        # Agrupar por query y contar URLs únicas
        cannibal = (
            top20.group_by('query')
            .agg([
                pl.col('page').n_unique().alias('url_count'),
                pl.col('clicks').sum().alias('total_clicks'),
                pl.col('impressions').sum().alias('total_impressions'),
                pl.col('page').alias('all_urls')
            ])
            .filter(pl.col('url_count') > 1)
        )
        
        total_queries = df_actual.select(pl.col('query').n_unique()).to_numpy().item()
        cannibal_count = cannibal.height
        
        rate = (cannibal_count / total_queries * 100) if total_queries > 0 else 0
        
        # Ordenar por el criterio especificado y limitar número de queries
        sort_column = 'total_impressions' if sort_by == 'impressions' else 'total_clicks'
        cannibal_df = cannibal.sort(sort_column, descending=True).head(max_queries)
        
        # Limitar URLs por query y unirlas
        def limit_urls(urls_list):
            """Limita URLs a max_urls_per_query y une con |"""
            unique_urls = list(dict.fromkeys(urls_list))[:max_urls_per_query]
            return '|'.join(unique_urls)
        
        # Aplicar límite de URLs
        urls_list = cannibal_df['all_urls'].to_list()
        limited_urls = [limit_urls(urls) for urls in urls_list]
        
        cannibal_df = cannibal_df.with_columns([
            pl.Series('urls', limited_urls)
        ]).drop('all_urls')
        
        status = 'ok' if rate <= self.targets.target_cannibalization else 'critical'
        
        return {
            'cannibalized_queries': cannibal_count,
            'total_queries': total_queries,
            'cannibalization_rate': round(rate, 2),
            'target': self.targets.target_cannibalization,
            'status': status,
            'cannibal_df': cannibal_df
        }
    
    def analyze(self) -> Dict:
        """Análisis completo de recovery."""
        return {
            'metricas_impacto': {
                'trafico_organico': self.calculate_traffic_recovery(),
                'impresiones_transaccionales': self.calculate_transactional_impressions(),
                'nonbranded_coverage': self.calculate_nonbranded_coverage(),
                'urls_optimizadas': self.calculate_url_optimization()
            },
            'indicadores_salud': {
                'top10_coverage': self.calculate_top10_coverage(),
                'cannibalization': self.detect_cannibalization()
            }
        }
    
    def print_dashboard(self):
        """Imprime dashboard de recovery."""
        results = self.analyze()
        
        print("=" * 60)
        print("SEO RECOVERY DASHBOARD")
        print("=" * 60)
        
        print("\n--- MÉTRICAS DE IMPACTO ---")
        
        trafico = results['metricas_impacto']['trafico_organico']
        if trafico.get('status') != 'no_target':
            print(f"Tráfico Orgánico:")
            print(f"  Target: {trafico['target']:,}")
            print(f"  Actual: {trafico['actual']:,}")
            print(f"  Progreso: {trafico['progress_pct']}%")
            print(f"  Status: {trafico['status']}")
        
        trans = results['metricas_impacto']['impresiones_transaccionales']
        if trans.get('status') != 'no_keywords':
            print(f"\nImpresiones Transaccionales:")
            print(f"  Base: {trans['base']:,}")
            print(f"  Actual: {trans['actual']:,}")
            print(f"  Variación: {trans['variation']['pct']:+.2f}%")
        
        nb = results['metricas_impacto']['nonbranded_coverage']
        print(f"\nCobertura Non-Branded:")
        print(f"  Recovery: {nb['recovery_pct']}%")
        print(f"  Status: {nb['status']}")
        
        urls = results['metricas_impacto']['urls_optimizadas']
        if urls.get('status') != 'no_target':
            print(f"\nURLs Optimizadas:")
            print(f"  Progreso: {urls['progress_pct']}% ({urls['urls_optimizadas']}/{urls['total_urls']})")
        
        print("\n--- INDICADORES DE SALUD ---")
        
        top10 = results['indicadores_salud']['top10_coverage']
        print(f"TOP 10 Coverage:")
        print(f"  Actual: {top10['coverage_pct']}%")
        print(f"  Target: {top10['target']}%")
        print(f"  Status: {top10['status']}")
        
        cann = results['indicadores_salud']['cannibalization']
        print(f"\nCannibalización:")
        print(f"  Rate: {cann['cannibalization_rate']}%")
        print(f"  Target: <{cann['target']}%")
        print(f"  Status: {cann['status']}")
