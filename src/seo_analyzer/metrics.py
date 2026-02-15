"""
Métricas avanzadas para SEOAnalyzer.
"""
from typing import Dict, List, Optional, Tuple, Union
import polars as pl
from polars import DataFrame, Series


class MetricsCalculator:
    """Calculador de métricas avanzadas."""
    
    @staticmethod
    def weighted_avg(series: Series, weights: Series) -> float:
        """Calcula promedio ponderado."""
        if series.len() == 0 or weights.len() == 0:
            return 0.0
        
        total_weight = weights.sum()
        if isinstance(total_weight, (int, float)):
            if total_weight == 0:
                return 0.0
        else:
            total_weight = total_weight.to_numpy().item()
            if total_weight == 0:
                return 0.0
        
        weighted_sum = (series * weights).sum()
        if not isinstance(weighted_sum, (int, float)):
            weighted_sum = weighted_sum.to_numpy().item()
        
        if isinstance(total_weight, (int, float)):
            return weighted_sum / total_weight
        return float(weighted_sum / total_weight)
    
    @staticmethod
    def std(series: Series) -> float:
        """Calcula desviación estándar."""
        if series.len() < 2:
            return 0.0
        result = series.std()
        if isinstance(result, (int, float)):
            return float(result)
        return result.to_numpy().item()
    
    @staticmethod
    def median(series: Series) -> float:
        """Calcula mediana."""
        if series.len() == 0:
            return 0.0
        result = series.median()
        if isinstance(result, (int, float)):
            return float(result)
        return result.to_numpy().item()
    
    @staticmethod
    def percentile(series: Series, q: float) -> float:
        """Calcula percentil."""
        if series.len() == 0:
            return 0.0
        result = series.quantile(q / 100)
        if isinstance(result, (int, float)):
            return float(result)
        return result.to_numpy().item()
    
    @staticmethod
    def summary_by_group(
        df: DataFrame,
        group_by: str,
        metrics: List[str] = None,
        weight_col: str = 'impressions',
        n: Optional[int] = None,
        sort_by: Optional[str] = None
    ) -> DataFrame:
        """Genera resumen con métricas avanzadas por grupo.

        Args:
            df: DataFrame con datos
            group_by: Columna a agrupar (query, page, grupo, etc.)
            metrics: Lista de métricas a calcular (default: clicks, impressions, ctr, position)
            weight_col: Columna para ponderación
            n: Número máximo de filas a retornar (default: None = todas)
            sort_by: Columna para ordenar (default: None = sin ordenar)

        Returns:
            DataFrame con métricas por grupo
        """
        if metrics is None:
            metrics = ['clicks', 'impressions', 'ctr', 'position']

        agg_exprs = []

        if 'clicks' in metrics:
            agg_exprs.extend([
                pl.col('clicks').sum().alias('sum_clicks'),
                pl.col('clicks').mean().alias('avg_clicks'),
                pl.col('clicks').median().alias('median_clicks'),
                pl.col('clicks').std().alias('std_clicks'),
                pl.col('clicks').count().alias('count_clicks')
            ])

        if 'impressions' in metrics:
            agg_exprs.append(pl.col('impressions').sum().alias('sum_impressions'))

        if 'ctr' in metrics:
            agg_exprs.append(pl.col('ctr').mean().round(2).alias('avg_ctr'))
        
        if 'position' in metrics:
            agg_exprs.extend([
                pl.col('position').mean().round(2).alias('avg_position'),
                pl.col('position').median().round(2).alias('median_position'),
                pl.col('position').std().alias('std_position')
            ])

        result = df.group_by(group_by).agg(agg_exprs)

        if sort_by and sort_by in result.columns:
            result = result.sort(sort_by, descending=True)

        if n is not None and n > 0:
            result = result.head(n)

        return result
    
    @staticmethod
    def summary_totals(
        df: DataFrame,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """Genera resumen de totales para un DataFrame.
        
        Args:
            df: DataFrame con datos
            metrics: Lista de métricas a calcular
        
        Returns:
            Dict con métricas totales
        """
        if metrics is None:
            metrics = ['clicks', 'impressions', 'ctr', 'position']
        
        result = {}
        
        if 'clicks' in metrics:
            result['sum_clicks'] = df.select(pl.col('clicks').sum()).to_numpy().item()
        
        if 'impressions' in metrics:
            result['sum_impressions'] = df.select(pl.col('impressions').sum()).to_numpy().item()
        
        if 'ctr' in metrics:
            result['avg_ctr'] = round(df.select(pl.col('ctr').mean()).to_numpy().item(), 2)
        
        if 'position' in metrics:
            result['avg_position'] = round(df.select(pl.col('position').mean()).to_numpy().item(), 2)
        
        return result
    
    @staticmethod
    def compare_periods(
        df1: DataFrame,
        df2: DataFrame,
        group_by: str = 'query',
        metrics: List[str] = None,
        n: Optional[int] = None,
        sort_by: Optional[str] = None
    ) -> DataFrame:
        """Compara dos períodos y calcula variaciones.

        Args:
            df1: DataFrame período 1
            df2: DataFrame período 2
            group_by: Columna a agrupar
            metrics: Métricas a comparar
            n: Número máximo de filas a retornar (default: None = todas)
            sort_by: Columna para ordenar por variación absoluta (ej: 'clicks_var_abs')

        Returns:
            DataFrame con comparación
        """
        if metrics is None:
            metrics = ['clicks', 'impressions']

        agg1 = df1.group_by(group_by).agg([
            pl.col(m).sum().alias(f'{m}_p1') for m in metrics
        ])

        agg2 = df2.group_by(group_by).agg([
            pl.col(m).sum().alias(f'{m}_p2') for m in metrics
        ])

        merged = agg1.join(agg2, on=group_by, how='full').fill_null(0)

        for m in metrics:
            merged = merged.with_columns([
                (pl.col(f'{m}_p2') - pl.col(f'{m}_p1')).alias(f'{m}_var_abs'),
                (
                    (pl.col(f'{m}_p2') - pl.col(f'{m}_p1')) /
                    pl.col(f'{m}_p1').replace(0, 1).replace(0, 1) * 100
                ).alias(f'{m}_var_pct')
            ])

        if sort_by and sort_by in merged.columns:
            merged = merged.sort(sort_by, descending=True)

        if n is not None and n > 0:
            merged = merged.head(n)

        return merged
    
    @staticmethod
    def top_n_by_variation(
        df1: DataFrame,
        df2: DataFrame,
        group_by: str = 'query',
        metric: str = 'clicks',
        n: int = 25
    ) -> DataFrame:
        """Obtiene top N elementos con mayor variación.
        
        Args:
            df1: DataFrame período 1
            df2: DataFrame período 2
            group_by: Columna a agrupar
            metric: Métrica a comparar
            n: Número de resultados
        
        Returns:
            DataFrame con top variaciones
        """
        agg1 = df1.group_by(group_by).agg(
            pl.col(metric).sum().alias(f'{metric}_p1')
        )
        
        agg2 = df2.group_by(group_by).agg(
            pl.col(metric).sum().alias(f'{metric}_p2')
        )
        
        merged = agg1.join(agg2, on=group_by, how='full').fill_null(0)
        
        merged = merged.with_columns([
            (pl.col(f'{metric}_p2') - pl.col(f'{metric}_p1')).alias('variacion'),
            (pl.col(f'{metric}_p2') - pl.col(f'{metric}_p1')).abs().alias('variacion_abs')
        ])
        
        return merged.sort('variacion_abs', descending=True).head(n)
    
    @staticmethod
    def distribution_summary(
        df: DataFrame,
        col: str,
        value_col: str = 'clicks'
    ) -> Dict:
        """Genera resumen de distribución por categoría.
        
        Args:
            df: DataFrame
            col: Columna categórica
            value_col: Columna de valor
        
        Returns:
            Dict con distribución
        """
        total = df.select(pl.col(value_col).sum()).to_numpy().item()
        
        dist = df.group_by(col).agg(
            pl.col(value_col).sum().alias('total'),
            pl.col(value_col).count().alias('count')
        )
        
        dist = dist.with_columns(
            (pl.col('total') / total * 100).alias('share_pct')
        )
        
        return {
            'total': total,
            'distribution': dist.sort('total', descending=True),
            'categories': dist.height
        }
