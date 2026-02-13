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

## Requisitos

- Python 3.11+
- polars >= 0.20.0

## Licencia

MIT
