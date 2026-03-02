// ==============================================================================
// 🛰️ ARKHE-1: PLANETARY NOOSPHERE & ENTROPY MAPPING
// ==============================================================================
// Motor Analítico: Google Earth Engine
// Paradigma: Termodinâmica da Informação (C + F = 1)
// ==============================================================================

// 1. ANCORAGEM FÍSICA E TOPOLÓGICA (Camadas de Reconhecimento Arkhe)
var CLA_LON_LAT = [-44.3966, -2.3155]; // Centro de Lançamento de Alcântara
var ALCANTARA_NODE = ee.Geometry.Point(CLA_LON_LAT);
var EQUATOR_LINE = ee.Geometry.LineString([[-180, 0], [180, 0]]);

// 2. FUNÇÃO DE ESTADO TERMODINÂMICO
// Normaliza as luzes noturnas (0-63) para a escala de Flutuação F [0, 1]
// e calcula a Coerência Residual C = 1 - F.
function computeThermodynamicState(img) {
  // Tempo T (Anos desde o início da observação base)
  var year = ee.Date(img.get('system:time_start')).get('year').subtract(1991);
  var timeBand = ee.Image(year).float().rename('time');

  // Flutuação Antropogênica F (Entropia local)
  var fluctuation_F = img.select('stable_lights').divide(63.0).rename('entropy_F');

  // Coerência Natural C (Estado fundamental)
  var coherence_C = ee.Image(1.0).subtract(fluctuation_F).rename('coherence_C');

  return timeBand.addBands(fluctuation_F)
                 .addBands(coherence_C)
                 .copyProperties(img, ['system:time_start']);
}

// 3. AQUISIÇÃO E MAPEAMENTO DO HIPERGRAFO
var noosphereCollection = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS')
  .map(computeThermodynamicState);

// 4. CÁLCULO DO GRADIENTE ENTROPICO (∇F)
// Calcula o ajuste linear da entropia ao longo do tempo.
var entropyTrend = noosphereCollection
  .select(['time', 'entropy_F'])
  .reduce(ee.Reducer.linearFit());

// ==============================================================================
// 🎨 RENDERIZAÇÃO E CAMADAS DE RECONHECIMENTO (VISUALIZAÇÃO)
// ==============================================================================

// Definir o fundo do mapa como Escuro (Vácuo)
Map.setOptions('SATELLITE');
Map.centerObject(ALCANTARA_NODE, 4); // Focar a câmera na Base Arkhe

// CAMADA 1: Gradiente Entrópico Global (A Tendência)
// Vermelho: Aceleração da Entropia (Crescimento urbano / Perda de C)
// Azul: Desaceleração da Entropia (Retorno ao estado de coerência)
// Verde: Entropia de base estabilizada (Offset)
var trendVis = {
  min: 0,
  max: [0.03, 0.8, -0.03], // Parâmetros calibrados para a escala normalizada [0, 1]
  bands: ['scale', 'offset', 'scale']
};
Map.addLayer(entropyTrend, trendVis, 'Arkhe-1: Gradiente Entrópico (∇F)');

// CAMADA 2: O Filtro Áureo (Anomalias Topológicas)
// Isola apenas as regiões onde a entropia atual ultrapassou a Proporção Áurea (F > 0.618)
var goldenRatioMask = noosphereCollection.limit(1, 'system:time_start', false).first()
  .select('entropy_F').gt(0.618);
Map.addLayer(goldenRatioMask.updateMask(goldenRatioMask),
  {palette: ['FFD700']}, 'Arkhe-1: Máscara de Ruptura Áurea (F > 0.618)', false);

// CAMADA 3: Estilingue Termodinâmico (O Equador)
Map.addLayer(EQUATOR_LINE, {color: '00FFFF', strokeWidth: 1}, 'Linha Equatorial (Estilingue)');

// CAMADA 4: Nó Zero (Base de Alcântara)
Map.addLayer(ALCANTARA_NODE, {color: 'FF00FF'}, 'Nó Zero: Base de Alcântara (CLA)');

print('📡 Telemetria Arkhe-1 Online: Renderizando Coerência e Flutuação da Noosfera.');
