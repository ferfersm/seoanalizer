---
name: seo-analyzer
description: Analiza datos de Google Search Console para tracking SEO y recovery post-migración. Usa cuando el usuario menciona análisis SEO, GSC, tráfico orgánico, recovery de tráfico, comparación de períodos, métricas de posicionamiento, o canibalización de keywords.
license: MIT
compatibility: Python 3.11+, polars
metadata:
  author: SEO Technical Team
  version: "1.4.4"
---

# seo_analyzer Skill

Módulo de análisis SEO de alto rendimiento usando Polars para datos de Google Search Console.

## Cuándo Usar Este Skill

Usa este skill cuando:
- El usuario menciona análisis de datos de Google Search Console (GSC)
- Necesitas analizar tráfico orgánico y métricas de posicionamiento
- El usuario quiere comparar períodos (antes/después de migración)
- Tracking de recovery post-migración de sitio web
- Detectar canibalización de keywords
- Analizar keywords por categorías
- Calcular métricas avanzadas

## Prerequisites

1. **Datos requeridos**: DataFrame de Polars con columnas:
   - `query` - Palabras clave (str)
   - `page` - URL de la página (str)
   - `date` - Fecha (datetime o str en formato ISO)
   - `clicks` - Clics (int)
   - `impressions` - Impresiones (int)
   - `ctr` - CTR (float, entre 0 y 1, o 0-100)
   - `position` - Posición (float)

2. **Tipos de datos**: El módulo automáticamente normaliza los tipos:
   - `clicks` e `impressions` → Int64 (enteros)
   - `ctr` y `position` → Float64 (2 decimales)
   - `date` → Datetime (si viene como string)
   - `query` → lowercase (minúsculas)

3. **Entorno**: Python 3.11+ con polars instalado

## Quick Start

```python
!pip install git+https://github.com/ferfersm/seoanalizer.git

import seo_analyzer
from seo_analyzer import SEOAnalyzer
import polars as pl

df = pl.read_csv('datos-gsc.csv')
analyzer = SEOAnalyzer(df)
analyzer.configure(
    periodo_base=('2025-01-01', '2025-03-31'),
    periodo_actual=('2025-04-01', '2025-06-30'),
    grupos={'Grupo': ['keyword1', 'keyword2']},
    brand_keywords=['marca'],
    kpi_keywords=['reserva', 'hora']
)
results = analyzer.analyze_all()
analyzer.print_summary()
```

## Errores Comunes

- "ModuleNotFoundError": Instalar con `pip install git+https://github.com/...`
- "Períodos no configurados": Ejecutar `configure()` con `periodo_base` y `periodo_actual`
- "No hay datos": Verificar formato de fechas (YYYY-MM-DD)

## Notas de Formato

- Todas las métricas de **CTR** y **position** se redondean automáticamente a **2 decimales**
- Esto asegura consistencia en los outputs y facilita la exportación a Sheets

## Métodos Disponibles (v1.4.4)

### SEOAnalyzer
| Método | Descripción |
|--------|-------------|
| `from_csv(path)` | Carga datos desde CSV (classmethod) |
| `from_parquet(path)` | Carga datos desde Parquet (classmethod) |
| `configure(...)` | Configura períodos, grupos, brand/KPIs |
| `filter_by_date(start, end)` | Filtra DataFrame por rango de fechas |
| `get_periodo('base' \| 'actual')` | Obtiene datos de un período configurado |
| `analyze_queries(periodo, n=100, sort_by='sum_clicks')` | Análisis agregado por query (limitado) |
| `get_top10_queries(periodo, n=100, sort_by='clicks')` | Queries con position <= 10 (limitado) |
| `analyze_urls(periodo, n=100, sort_by='sum_clicks')` | Análisis agregado por URL (limitado) |
| `analyze_totals(periodo)` | Métricas totales del período |
| `analyze_brand_vs_nonbrand(periodo)` | Brand vs non-branded |
| `analyze_grupos(periodo)` | Análisis por grupos de keywords |
| `analyze_kpis(periodo)` | Análisis de keywords KPIs |
| `compare_periods()` | Compara período base vs actual |
| `top_queries_variation(n)` | Top N queries con mayor variación |
| `analyze_all()` | Ejecuta análisis completo |
| `print_summary()` | Imprime resumen ejecutivo en consola |
| `generate_subsets(df)` | Genera subconjuntos por categorías |
| `summarize_dataframes(dfs)` | Resume DataFrames con métricas |
| `compare_summaries(df1, df2)` | Compara resúmenes entre períodos |
| `create_comparison_df(curr, prev, labels, metric)` | Crea DataFrame comparativo |
| `analyze_subdomains(prev, curr, patterns)` | Análisis por subdominios |
| `top_n_queries_by_variation(prev, curr, n, metric)` | Top queries con variación |
| `traffic_distribution_by_keyword_category(df)` | Distribución por categoría |
| `traffic_distribution_by_subdomain(df)` | Distribución por subdominio |
| `resumen_kw(kw, periodo, col)` | Seguimiento temporal de keywords (v1.4.3) |

### Funciones Utilitarias
| Función | Descripción |
|---------|-------------|
| `mostrar_df(df, formato)` | Visualización/exportación de DataFrames (v1.4.3) |

### ComparisonAnalyzer
| Método | Descripción |
|--------|-------------|
| `compare_totals(p1, p2)` | Compara totales entre dos períodos |
| `compare_by_group(p1, p2, col)` | Compara por grupo entre períodos |
| `compare_brand_nonbrand(p1, p2)` | Compara brand vs non-brand |
| `top_variation_queries(p1, p2, n)` | Top queries con mayor variación |
| `full_comparison(p1, p2)` | Comparación completa de períodos |

### RecoveryAnalyzer
| Método | Descripción |
|--------|-------------|
| `apply_classifications()` | Aplica clasificaciones de grupo, brand, KPI |
| `calculate_traffic_recovery()` | Recovery de tráfico |
| `calculate_transactional_impressions()` | Impresiones transaccionales |
| `calculate_nonbranded_coverage()` | Cobertura non-branded |
| `calculate_url_optimization()` | Optimización URLs |
| `calculate_top10_coverage(max_queries=100, max_urls_per_query=10, sort_by='clicks')` | % keywords en top 10 + DataFrame limitado |
| `detect_cannibalization(max_queries=100, max_urls_per_query=10, sort_by='impressions')` | Detecta canibalización + DataFrame |
| `analyze()` | Análisis completo de recovery |
| `print_dashboard()` | Imprime dashboard de recovery |

### MetricsCalculator (Estático)
| Método | Descripción |
|--------|-------------|
| `weighted_avg(series, weights)` | Promedio ponderado |
| `std(series)` | Desviación estándar |
| `median(series)` | Mediana |
| `percentile(series, q)` | Percentil |
| `summary_by_group(df, group_by, metrics, n, sort_by)` | Resumen por grupo con métricas avanzadas |
| `summary_totals(df, metrics)` | Resumen de totales |
| `compare_periods(df1, df2, group_by, metrics, n, sort_by)` | Compara dos períodos |
| `top_n_by_variation(df1, df2, group_by, metric, n)` | Top N por variación |
| `distribution_summary(df, col, value_col)` | Distribución por categoría |

---

## Ejemplo Completo de Uso

```python
"""
Script de análisis SEO completo - seo_analyzer v1.4.4
Compatible con Google Colab y Jupyter Notebooks
"""

# ==============================================================================
# 1. IMPORTS Y CONFIGURACIÓN
# ==============================================================================
import polars as pl
from seo_analyzer import (
    SEOAnalyzer, 
    RecoveryTargets, 
    RecoveryAnalyzer,
    mostrar_df  # NUEVO en v1.4.3
)
from IPython.display import display

# Configurar Polars
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(-1)

# ==============================================================================
# 2. CARGA DE DATOS
# ==============================================================================
df = pl.read_csv('datos-gsc.csv')

# ==============================================================================
# 3. CONFIGURACIÓN
# ==============================================================================
grupos = {
    "Marca": ['transbank', 'tbk'],
    "Productos": ['webpay', 'onepay']
}

recovery_targets = RecoveryTargets(
    target_trafico=50000,
    target_top10_coverage=40.0,
    target_cannibalization=15.0
)

analyzer = SEOAnalyzer(df)
analyzer.configure(
    cliente='MiCliente',
    periodo_base=('2025-01-01', '2025-01-31'),
    periodo_actual=('2025-02-01', '2025-02-28'),
    grupos=grupos,
    brand_keywords=['transbank', 'tbk'],
    kpi_keywords=['reserva', 'hora'],
    recovery_targets=recovery_targets
)

# ==============================================================================
# 4. NUEVO: mostrar_df() (v1.4.3)
# ==============================================================================
# Mostrar en CSV (para copiar a Sheets)
mostrar_df(df.head(10), formato='csv')

# Mostrar como tabla (para Colab)
mostrar_df(df.head(10), formato='display')

# ==============================================================================
# 5. NUEVO: resumen_kw() (v1.4.3)
# ==============================================================================
# Seguimiento de keyword específica
evolucion = analyzer.resumen_kw(
    kw='transbank',
    periodo='month',  # 'month' o 'day'
    exact_match=True
)

# Seguimiento de TODO el tráfico
total = analyzer.resumen_kw(kw=None, periodo='month')

# Seguimiento por URL
urls = analyzer.resumen_kw(col='page', kw='/home', periodo='month')

# ==============================================================================
# 6. ANÁLISIS BÁSICO
# ==============================================================================
# Totales por período
totales = analyzer.analyze_totals('actual')

# Top queries (con límites)
queries = analyzer.analyze_queries('actual', n=50, sort_by='sum_clicks')

# Top URLs (con límites)
urls = analyzer.analyze_urls('actual', n=50, sort_by='sum_impressions')

# Queries en TOP 10 (position <= 10)
top10 = analyzer.get_top10_queries('actual', n=100, sort_by='clicks')

# ==============================================================================
# 7. ANÁLISIS POR CATEGORÍAS
# ==============================================================================
# Por grupos definidos
grupos = analyzer.analyze_grupos('actual')

# Brand vs Non-brand
brand = analyzer.analyze_brand_vs_nonbrand('actual')

# KPIs
kpis = analyzer.analyze_kpis('actual')

# ==============================================================================
# 8. COMPARACIONES
# ==============================================================================
# Comparar períodos
comparacion = analyzer.compare_periods()

# Top variaciones
variaciones = analyzer.top_queries_variation(25)

# ==============================================================================
# 9. MÉTODOS AVANZADOS
# ==============================================================================
base = analyzer.get_periodo('base')
actual = analyzer.get_periodo('actual')

# Subsets y resúmenes
subsets = analyzer.generate_subsets(actual)
summary = analyzer.summarize_dataframes(subsets)

# Comparación de resúmenes
comparison = analyzer.compare_summaries(
    analyzer.summarize_dataframes(analyzer.generate_subsets(base)),
    summary
)

# Distribución por categoría
dist_cat = analyzer.traffic_distribution_by_keyword_category(actual)

# Distribución por subdominio
dist_sub = analyzer.traffic_distribution_by_subdomain(actual)

# ==============================================================================
# 10. RECOVERY ANALYZER
# ==============================================================================
recovery = RecoveryAnalyzer(df, analyzer.config, recovery_targets)
recovery.apply_classifications()

# Métricas de recovery
trafico = recovery.calculate_traffic_recovery()
trans = recovery.calculate_transactional_impressions()
nb = recovery.calculate_nonbranded_coverage()
urls = recovery.calculate_url_optimization()

# TOP 10 Coverage (con límites)
top10_cov = recovery.calculate_top10_coverage(
    max_queries=100,
    max_urls_per_query=10,
    sort_by='clicks'
)
mostrar_df(top10_cov['top10_df'], formato='csv')

# Canibalización (con límites y position)
cann = recovery.detect_cannibalization(
    max_queries=100,
    max_urls_per_query=10,
    sort_by='impressions'
)
mostrar_df(cann['cannibal_df'], formato='csv')

# Dashboard completo
recovery.print_dashboard()

# ==============================================================================
# 11. ANÁLISIS COMPLETO
# ==============================================================================
results = analyzer.analyze_all()
analyzer.print_summary()
```
