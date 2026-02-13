"""
ComparisonAnalyzer - Análisis comparativo de períodos.
"""
from typing import Dict, List, Optional, Tuple
import polars as pl

from .config import SEOConfig
from .utils import filter_by_date_range, compile_pattern


class ComparisonAnalyzer:
    """Analizador de comparación de períodos.
    
    Uso:
        comp = ComparisonAnalyzer(df, config)
        comp.compare_periods(periodo_1, periodo_2)
    """
    
    def __init__(self, df: pl.DataFrame, config: SEOConfig):
        """Inicializa con DataFrame y configuración."""
        self.df = df
        self.config = config
    
    def compare_totals(
        self,
        periodo_1: Tuple[str, str],
        periodo_2: Tuple[str, str]
    ) -> Dict:
        """Compara totales entre dos períodos."""
        df1 = filter_by_date_range(self.df, periodo_1[0], periodo_1[1])
        df2 = filter_by_date_range(self.df, periodo_2[0], periodo_2[1])
        
        metrics = ['clicks', 'impressions', 'ctr', 'position']
        
        result = {}
        
        for metric in metrics:
            if metric in ['clicks', 'impressions']:
                val1 = df1.select(pl.col(metric).sum()).to_numpy().item()
                val2 = df2.select(pl.col(metric).sum()).to_numpy().item()
            else:
                val1 = df1.select(pl.col(metric).mean()).to_numpy().item()
                val2 = df2.select(pl.col(metric).mean()).to_numpy().item()
            
            var_abs = val2 - val1
            var_pct = ((val2 - val1) / val1 * 100) if val1 != 0 else 0
            
            result[metric] = {
                'periodo_1': val1,
                'periodo_2': val2,
                'var_abs': var_abs,
                'var_pct': round(var_pct, 2)
            }
        
        return result
    
    def compare_by_group(
        self,
        periodo_1: Tuple[str, str],
        periodo_2: Tuple[str, str],
        group_col: str = 'grupo'
    ) -> Dict:
        """Compara por grupo entre dos períodos."""
        if not self.config.grupos:
            return {}
        
        df1 = filter_by_date_range(self.df, periodo_1[0], periodo_1[1])
        df2 = filter_by_date_range(self.df, periodo_2[0], periodo_2[1])
        
        result = {}
        
        for grupo in self.config.grupos.keys():
            df1_g = df1.filter(pl.col(group_col) == grupo)
            df2_g = df2.filter(pl.col(group_col) == grupo)
            
            clicks1 = df1_g.select(pl.col('clicks').sum()).to_numpy().item()
            clicks2 = df2_g.select(pl.col('clicks').sum()).to_numpy().item()
            
            imp1 = df1_g.select(pl.col('impressions').sum()).to_numpy().item()
            imp2 = df2_g.select(pl.col('impressions').sum()).to_numpy().item()
            
            pos1 = df1_g.select(pl.col('position').mean()).to_numpy().item()
            pos2 = df2_g.select(pl.col('position').mean()).to_numpy().item()
            
            result[grupo] = {
                'clicks': {
                    'periodo_1': clicks1,
                    'periodo_2': clicks2,
                    'var_pct': round(((clicks2 - clicks1) / clicks1 * 100) if clicks1 != 0 else 0, 2)
                },
                'impressions': {
                    'periodo_1': imp1,
                    'periodo_2': imp2,
                    'var_pct': round(((imp2 - imp1) / imp1 * 100) if imp1 != 0 else 0, 2)
                },
                'position': {
                    'periodo_1': round(pos1, 1),
                    'periodo_2': round(pos2, 1),
                    'var': round(pos2 - pos1, 1)
                }
            }
        
        return result
    
    def compare_brand_nonbrand(
        self,
        periodo_1: Tuple[str, str],
        periodo_2: Tuple[str, str]
    ) -> Dict:
        """Compara brand vs non-brand entre períodos."""
        df1 = filter_by_date_range(self.df, periodo_1[0], periodo_1[1])
        df2 = filter_by_date_range(self.df, periodo_2[0], periodo_2[1])
        
        result = {}
        
        for label, col_filter in [('brand', True), ('non_brand', False)]:
            df1_b = df1.filter(pl.col('es_brand') == col_filter)
            df2_b = df2.filter(pl.col('es_brand') == col_filter)
            
            clicks1 = df1_b.select(pl.col('clicks').sum()).to_numpy().item()
            clicks2 = df2_b.select(pl.col('clicks').sum()).to_numpy().item()
            
            imp1 = df1_b.select(pl.col('impressions').sum()).to_numpy().item()
            imp2 = df2_b.select(pl.col('impressions').sum()).to_numpy().item()
            
            result[label] = {
                'clicks': {
                    'periodo_1': clicks1,
                    'periodo_2': clicks2,
                    'var_pct': round(((clicks2 - clicks1) / clicks1 * 100) if clicks1 != 0 else 0, 2)
                },
                'impressions': {
                    'periodo_1': imp1,
                    'periodo_2': imp2,
                    'var_pct': round(((imp2 - imp1) / imp1 * 100) if imp1 != 0 else 0, 2)
                }
            }
        
        return result
    
    def top_variation_queries(
        self,
        periodo_1: Tuple[str, str],
        periodo_2: Tuple[str, str],
        metric: str = 'clicks',
        n: int = 25
    ) -> pl.DataFrame:
        """Top queries con mayor variación."""
        df1 = filter_by_date_range(self.df, periodo_1[0], periodo_1[1])
        df2 = filter_by_date_range(self.df, periodo_2[0], periodo_2[1])
        
        col_query = self.config.columnas.get('query', 'query')
        
        agg1 = df1.group_by(col_query).agg(
            pl.col(metric).sum().alias(f'{metric}_p1')
        )
        
        agg2 = df2.group_by(col_query).agg(
            pl.col(metric).sum().alias(f'{metric}_p2')
        )
        
        merged = agg1.join(agg2, on=col_query, how='full').fill_null(0)
        
        merged = merged.with_columns([
            (pl.col(f'{metric}_p2') - pl.col(f'{metric}_p1')).alias('variacion'),
            (pl.col(f'{metric}_p2') - pl.col(f'{metric}_p1')).abs().alias('variacion_abs')
        ])
        
        return merged.sort('variacion_abs', descending=True).head(n)
    
    def full_comparison(
        self,
        periodo_1: Tuple[str, str],
        periodo_2: Tuple[str, str]
    ) -> Dict:
        """Comparación completa."""
        from .utils import classify_by_keywords, classify_brand
        
        df_to_use = self.df
        
        if 'grupo' not in df_to_use.columns and self.config.grupos:
            df_to_use = classify_by_keywords(df_to_use, self.config.grupo_map, self.config.columnas.get('query', 'query'))
        
        if 'es_brand' not in df_to_use.columns and self.config.brand_keywords:
            df_to_use = classify_brand(df_to_use, self.config.brand_keywords, self.config.columnas.get('query', 'query'))
        
        temp_analyzer = ComparisonAnalyzer(df_to_use, self.config)
        
        return {
            'periodos': {
                'periodo_1': periodo_1,
                'periodo_2': periodo_2
            },
            'totales': temp_analyzer.compare_totals(periodo_1, periodo_2),
            'brand_nonbrand': temp_analyzer.compare_brand_nonbrand(periodo_1, periodo_2),
            'grupos': temp_analyzer.compare_by_group(periodo_1, periodo_2),
            'top_variation': temp_analyzer.top_variation_queries(periodo_1, periodo_2)
        }
