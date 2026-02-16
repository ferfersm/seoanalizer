# seo_analyzer

Módulo de análisis SEO de alto rendimiento usando Polars para datos de Google Search Console.

## Características

- Análisis de datos de Google Search Console
- Comparación de períodos (pre/post migración)
- Métricas avanzadas (promedio ponderado, mediana, desviación estándar)
- Clasificación por grupos de keywords
- Recovery post-migración
- Detección de canibalización
- Alto rendimiento con Polars
- **Nuevo en v1.4**: Control de tamaño de outputs y normalización automática de datos

## Instalación

```python
!pip install git+https://github.com/ferfersm/seoanalizer.git
```

## Uso Rápido

```python
import seo_analyzer
from seo_analyzer import SEOAnalyzer
import polars as pl

# Cargar datos
df = pl.read_csv('datos-gsc.csv')

# Inicializar y configurar
analyzer = SEOAnalyzer(df)
analyzer.configure(
    periodo_base=('2025-01-01', '2025-03-31'),
    periodo_actual=('2025-04-01', '2025-06-30'),
    grupos={'Grupo1': ['keyword1', 'keyword2']},
    brand_keywords=['tu marca'],
    kpi_keywords=['reserva', 'hora']
)

# Analizar
results = analyzer.analyze_all()
analyzer.print_summary()
```

## Novedades v1.4

### Control de Tamaño de Outputs
Los métodos que retornan DataFrames ahora incluyen parámetros `n` y `sort_by` para evitar outputs masivos:

```python
# Top 100 queries por clicks (default)
queries = analyzer.analyze_queries('actual')

# Top 50 por impressions
queries = analyzer.analyze_queries('actual', n=50, sort_by='sum_impressions')

# Top 100 URLs en TOP 10 por impressions
top10 = analyzer.get_top10_queries('actual', n=100, sort_by='impressions')
```

### Métricas Redondeadas
Todas las métricas de **CTR** y **position** se redondean automáticamente a **2 decimales** para mejor legibilidad.

### Normalización de Tipos de Datos
El módulo normaliza automáticamente los tipos de datos:
- `clicks`, `impressions` → Int64 (enteros)
- `ctr`, `position` → Float64 (decimales)
- `date` → Datetime
- `query` → lowercase

## Métodos Principales

### SEOAnalyzer

| Método | Descripción |
|--------|-------------|
| `analyze_queries(periodo, n=100, sort_by='sum_clicks')` | Análisis por query con límites |
| `get_top10_queries(periodo, n=100, sort_by='clicks')` | Queries en TOP 10 (position ≤ 10) |
| `analyze_urls(periodo, n=100, sort_by='sum_clicks')` | Análisis por URL |
| `analyze_totals(periodo)` | Métricas totales |
| `analyze_brand_vs_nonbrand(periodo)` | Análisis brand vs non-brand |
| `analyze_grupos(periodo)` | Análisis por grupos configurados |
| `analyze_kpis(periodo)` | Análisis de keywords KPIs |
| `compare_periods()` | Comparación período base vs actual |
| `top_queries_variation(n)` | Top N queries con mayor variación |

### RecoveryAnalyzer (para análisis post-migración)

```python
from seo_analyzer import RecoveryTargets, RecoveryAnalyzer

# Configurar targets
targets = RecoveryTargets(
    target_trafico=100000,
    target_top10_coverage=40.0,
    target_cannibalization=15.0
)

# Analizar recovery
recovery = RecoveryAnalyzer(df, analyzer.config, targets)
results = recovery.analyze()
recovery.print_dashboard()
```

## Requisitos de Datos

El DataFrame debe contener estas columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `query` | str | Keywords de búsqueda |
| `page` | str | URLs de las páginas |
| `date` | datetime/str | Fecha (formato: YYYY-MM-DD) |
| `clicks` | int | Número de clics |
| `impressions` | int | Número de impresiones |
| `ctr` | float | Click-through rate (0-100 o 0-1) |
| `position` | float | Posición promedio |

**Nota**: Los tipos de datos se normalizan automáticamente al cargar.

## Ejemplo Completo

```python
import polars as pl
from seo_analyzer import SEOAnalyzer, RecoveryTargets, RecoveryAnalyzer

# Cargar datos
df = pl.read_csv('gsc-data.csv')

# Configurar analizador
analyzer = SEOAnalyzer(df)
analyzer.configure(
    cliente='Mi Cliente',
    periodo_base=('2025-01-01', '2025-03-31'),
    periodo_actual=('2025-04-01', '2025-06-30'),
    grupos={
        'Producto A': ['keyword1', 'keyword2'],
        'Producto B': ['keyword3', 'keyword4']
    },
    brand_keywords=['mi marca'],
    kpi_keywords=['comprar', 'precio', 'cotizar']
)

# Análisis básico
results = analyzer.analyze_all()

# Top 50 queries con mayor caída
queries_caidas = analyzer.top_queries_variation(50)

# Recovery post-migración
targets = RecoveryTargets(
    target_trafico=50000,
    target_top10_coverage=35.0
)
recovery = RecoveryAnalyzer(df, analyzer.config, targets)
recovery.analyze()
```

## Requisitos

- Python 3.11+
- polars >= 0.20.0

## Licencia

MIT

## Nuevas Funcionalidades v1.4.3

### `mostrar_df()` - Visualización y Exportación
Función utilitaria para mostrar DataFrames en formato CSV (para copiar a Sheets) o display (para Colab):

```python
from seo_analyzer import mostrar_df

# Exportar a CSV para copiar a Sheets
mostrar_df(df, formato='csv')

# Mostrar como tabla en Colab/Jupyter
mostrar_df(df, formato='display')
```

### `resumen_kw()` - Seguimiento Temporal de Keywords
Método para analizar la evolución temporal de una keyword específica o de todo el tráfico:

```python
# Seguimiento de keyword específica por mes
evolucion = analyzer.resumen_kw(kw='transbank', periodo='month')

# Resumen de todo el tráfico (sin filtrar)
resumen_total = analyzer.resumen_kw(kw=None, periodo='day')

# Seguimiento por URL en lugar de query
urls = analyzer.resumen_kw(col='page', kw='/login', periodo='month')
```

**Parámetros:**
- `kw`: Keyword a filtrar (None = todo el DataFrame)
- `col`: Columna para filtrar ('query' o 'page')
- `periodo`: 'month' o 'day'
- `exact_match`: True para match exacto, False para contains

**Columnas del resultado:**
- `period`: Período (mes o día)
- `clicks`, `impressions`: Totales del período
- `ctr`: Calculado como clicks/impressions (float, 2 decimales)
- `avg_position`: Promedio de posiciones (float, 2 decimales)
- `variacion_*`: Diferencias vs período anterior (absolutas y %)

## Changelog

### v1.4.3
- **Fix crítico**: Manejo de DataFrames vacíos en todos los métodos
- **Fix**: CTR calculado correctamente como clicks/impressions
- **Fix**: Position como promedio simple (no ponderado)
- **Nuevo**: `mostrar_df()` para visualización/exportación
- **Nuevo**: `resumen_kw()` para seguimiento temporal de keywords

### v1.4.2
- Normalización automática de tipos de datos
- Documentación actualizada

### v1.4.1
- Redondeo de CTR y position a 2 decimales

### v1.4.0
- Parámetros `n` y `sort_by` para controlar tamaño de outputs
- Límites en DataFrames para evitar outputs masivos

### v1.3.x
- RecoveryAnalyzer con métricas de recovery
- Detección de canibalización
- Cobertura TOP 10
