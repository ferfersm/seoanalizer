---
name: seo-analyzer
description: Analiza datos de Google Search Console para tracking SEO y recovery post-migración. Usa cuando el usuario menciona análisis SEO, GSC, tráfico orgánico, recovery de tráfico, comparación de períodos, métricas de posicionamiento, o canibalización de keywords.
license: MIT
compatibility: Python 3.11+, polars
metadata:
  author: SEO Technical Team
  version: "1.2"
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
   - `query` - Palabras clave
   - `page` - URL de la página
   - `date` - Fecha
   - `clicks` - Clics
   - `impressions` - Impresiones
   - `ctr` - CTR
   - `position` - Posición

2. **Entorno**: Python 3.11+ con polars instalado

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

## Métodos Disponibles (v1.2)

### SEOAnalyzer
| Método | Descripción |
|--------|-------------|
| `analyze_queries(periodo)` | Análisis agregado por query |
| `get_top10_queries(periodo)` | Queries con position <= 10 |
| `analyze_urls(periodo)` | Análisis agregado por URL |
| `analyze_totals(periodo)` | Métricas totales del período |
| `analyze_brand_vs_nonbrand(periodo)` | Brand vs non-branded |
| `analyze_grupos(periodo)` | Análisis por grupos de keywords |
| `analyze_kpis(periodo)` | Análisis de keywords KPIs |
| `compare_periods()` | Compara período base vs actual |
| `top_queries_variation(n)` | Top N queries con mayor variación |
| `generate_subsets(df)` | Genera subconjuntos por categorías |
| `summarize_dataframes(dfs)` | Resume DataFrames con métricas |
| `compare_summaries(df1, df2)` | Compara resúmenes entre períodos |
| `create_comparison_df(curr, prev, labels, metric)` | Crea DataFrame comparativo |
| `analyze_subdomains(prev, curr, patterns)` | Análisis por subdominios |
| `top_n_queries_by_variation(prev, curr, n, metric)` | Top queries con variación |
| `traffic_distribution_by_keyword_category(df)` | Distribución por categoría |
| `traffic_distribution_by_subdomain(df)` | Distribución por subdominio |

### RecoveryAnalyzer
| Método | Descripción |
|--------|-------------|
| `calculate_traffic_recovery()` | Recovery de tráfico |
| `calculate_transactional_impressions()` | Impresiones transaccionales |
| `calculate_nonbranded_coverage()` | Cobertura non-branded |
| `calculate_url_optimization()` | Optimización URLs |
| `calculate_top10_coverage()` | % keywords en top 10 |
| `detect_cannibalization()` | Detecta canibalización |
