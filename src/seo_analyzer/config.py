"""
Configuración para SEOAnalyzer.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import unicodedata


def normalizar_texto(texto: str) -> str:
    """Normaliza texto: minúsculas, sin tildes."""
    if not texto:
        return ""
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                   if unicodedata.category(c) != 'Mn')
    return texto.lower().strip()


@dataclass
class RecoveryTargets:
    """Targets para análisis de recovery."""
    target_trafico: Optional[int] = None
    target_top10_coverage: float = 40.0
    target_cannibalization: float = 15.0
    target_optimizacion_urls: float = 90.0
    keywords_transaccionales: List[str] = field(default_factory=list)
    total_urls_sitio: Optional[int] = None
    urls_ya_optimizadas: List[str] = field(default_factory=list)


@dataclass
class SEOConfig:
    """Configuración principal para SEOAnalyzer."""
    
    cliente: str = "cliente_generico"
    
    periodo_base: Optional[Tuple[str, str]] = None
    periodo_actual: Optional[Tuple[str, str]] = None
    
    grupos: Dict[str, List[str]] = field(default_factory=dict)
    
    brand_keywords: List[str] = field(default_factory=list)
    
    kpi_keywords: List[str] = field(default_factory=list)
    
    columnas: Dict[str, str] = field(default_factory=lambda: {
        'fecha': 'date',
        'query': 'query',
        'page': 'page',
        'clics': 'clicks',
        'impresiones': 'impressions',
        'ctr': 'ctr',
        'posicion': 'position'
    })
    
    recovery_targets: Optional[RecoveryTargets] = None
    
    @property
    def grupo_map(self) -> Dict[str, str]:
        """Crea mapeo palabra -> nombre_grupo (normalizado)."""
        return {
            normalizar_texto(palabra): grupo
            for grupo, palabras in self.grupos.items()
            for palabra in palabras
        }
    
    @property
    def todas_palabras(self) -> List[str]:
        """Lista plana de todas las palabras de grupos."""
        return [
            normalizar_texto(palabra)
            for grupo in self.grupos.values()
            for palabra in grupo
        ]
    
    def obtener_grupo(self, texto: str) -> Optional[str]:
        """Determina el grupo de un texto."""
        texto_norm = normalizar_texto(texto)
        for grupo, palabras in self.grupos.items():
            for palabra in palabras:
                if normalizar_texto(palabra) in texto_norm:
                    return grupo
        return None
    
    def es_brand(self, texto: str) -> bool:
        """Verifica si es keyword de marca."""
        texto_norm = normalizar_texto(texto)
        return any(normalizar_texto(kw) in texto_norm for kw in self.brand_keywords)
    
    def es_kpi(self, texto: str) -> bool:
        """Verifica si es keyword importante (KPI)."""
        texto_norm = normalizar_texto(texto)
        return any(normalizar_texto(kw) in texto_norm for kw in self.kpi_keywords)
