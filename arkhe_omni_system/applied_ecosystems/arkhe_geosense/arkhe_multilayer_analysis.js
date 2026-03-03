// ARKHE-1: ANÁLISE MULTICAMADAS DE LUZES NOTURNAS
// Alinhado com a arquitetura de reconhecimento geográfico do sistema Arkhe-1
// Versão: 1.0 (Integração com Hipergrafo Anyônico)

// ============================================
// CAMADA 1: FUNDAÇÃO TEMPORAL (COERÊNCIA BASE)
// ============================================

/**
 * Adiciona múltiplas bandas temporais para análise de coerência.
 * Arkhe-1: Diferentes escalas de tempo revelam diferentes padrões de handover
 */
function createTimeBands(img) {
  var date = ee.Date(img.get('system:time_start'));

  // Tempo absoluto (para regressão linear) - ESCALA GROSSA
  var yearAbsolute = date.get('year').subtract(1991);

  // Tempo cíclico (para análise de sazonalidade) - ESCALA MÉDIA
  var monthOfYear = date.get('month');
  var dayOfYear = date.get('dayOfYear');

  // Tempo relativo (para coerência local) - ESCALA FINA
  var daysSinceEpoch = date.difference(ee.Date('1991-01-01'), 'day');

  // Fase harmônica (inspirada na estatística anyônica do Arkhe-1)
  var phi = 1.618; // Proporção áurea
  var harmonicPhase = daysSinceEpoch.mod(365).multiply(2 * Math.PI / 365);

  return img
    .addBands(ee.Image(yearAbsolute).float().rename('year_absolute'))
    .addBands(ee.Image(monthOfYear).byte().rename('month'))
    .addBands(ee.Image(dayOfYear).short().rename('day_of_year'))
    .addBands(ee.Image(daysSinceEpoch).double().rename('days_epoch'))
    .addBands(ee.Image(harmonicPhase).float().rename('harmonic_phase'))
    .addBands(ee.Image(phi).float().rename('phi_golden'));
}

// ============================================
// CAMADA 2: FONTES DE DADOS HÍBRIDAS (DMSP + VIIRS)
// ============================================

// Coleção 1: DMSP-OLS (histórico 1992-2013)
var dmspCollection = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS')
    .select('stable_lights')
    .filterDate('1992-01-01', '2014-01-01')
    .map(createTimeBands);

// Coleção 2: VIIRS-DNB (moderno 2012-presente)
var viirsCollection = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
    .select(['avg_rad'], ['stable_lights'])
    .filterDate('2012-01-01', '2026-01-01')
    .map(createTimeBands);

// Unificação das coleções (Arkhe-1: fusão de sensores)
var unifiedCollection = dmspCollection.merge(viirsCollection);

// ============================================
// CAMADA 3: MÁSCARAS GEOGRÁFICAS MULTICAMADAS
// ============================================

// Definição da área de estudo (Brasil)
var brazil = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')
    .filter(ee.Filter.eq('country_na', 'Brazil'));

// CAMADA URBANA: Áreas com alta densidade populacional
var population = ee.ImageCollection('CIESIN/GPWv411/GPW_Population_Density')
    .filterDate('2020-01-01')
    .select('population_density')
    .first();

var urbanMask = population.gt(500).rename('urban_zone');

// CAMADA RURAL: Áreas de baixa densidade e parques nacionais
var ruralMask = population.lt(50).rename('rural_zone');

// CAMADA COSTEIRA: Zonas de interface mar-terra
var distanceToCoast = ee.Image('NOAA/NGDC/ETOPO1').select('bedrock')
    .not().not().distance(1000); // Buffer costeiro 1km
var coastalMask = distanceToCoast.lt(1000).rename('coastal_zone');

// CAMADA DE INFRAESTRUTURA (rodovias, aeroportos)
var roads = ee.FeatureCollection('TIGER/2016/Roads');
var roadDistance = roads.distance(5000); // 5km buffer
var infrastructureMask = roadDistance.lt(5000).rename('infrastructure_zone');

// ============================================
// CAMADA 4: ANÁLISE DE COERÊNCIA MULTIVARIADA
// ============================================

/**
 * Calcula múltiplos indicadores de coerência para cada pixel.
 * Arkhe-1: O hipergrafo precisa de diferentes métricas de fase
 */
function calculateCoherenceMetrics(collection) {

  // 1. Regressão linear simples (tendência temporal)
  var linearTrend = collection.select(['year_absolute', 'stable_lights'])
      .reduce(ee.Reducer.linearFit());

  // 2. Regressão robusta (resistente a outliers)
  var robustTrend = collection.select(['year_absolute', 'stable_lights'])
      .reduce(ee.Reducer.robustLinearRegression(2));

  // 3. Análise de sazonalidade (por mês)
  var seasonalMean = collection.select(['month', 'stable_lights'])
      .reduce(ee.Reducer.mean().group({
        groupField: 0,
        groupName: 'month'
      }));

  // 4. Métrica de coerência Φ (inspirada no Arkhe-1)
  // Quanto menor a variância residual, maior a coerência
  var residuals = collection.select(['stable_lights'])
      .map(function(img) {
        var predicted = linearTrend.select('scale')
            .multiply(img.select('year_absolute'))
            .add(linearTrend.select('offset'));
        var residual = img.select('stable_lights').subtract(predicted).pow(2);
        return residual.rename('residual');
      });

  var coherencePhi = residuals.reduce(ee.Reducer.mean())
      .rename('phi_coherence')
      .multiply(-1) // Inverte para que maior = mais coerente
      .add(1);      // Normaliza para [0,1] aproximado

  return {
    trend: linearTrend,
    robust: robustTrend,
    seasonal: seasonalMean,
    coherence: coherencePhi
  };
}

// ============================================
// CAMADA 5: APLICAÇÃO DAS ANÁLISES POR REGIÃO
// ============================================

// Enriquecer coleção com máscaras zonais
var enrichedCollection = unifiedCollection
    .map(function(img) {
      return img
          .addBands(urbanMask.rename('mask_urban'))
          .addBands(ruralMask.rename('mask_rural'))
          .addBands(coastalMask.rename('mask_coastal'))
          .addBands(infrastructureMask.rename('mask_infra'));
    });

// Calcular métricas para cada zona aplicando as máscaras no momento do cálculo
var urbanMetrics = calculateCoherenceMetrics(
    enrichedCollection.map(function(img) { return img.updateMask(img.select('mask_urban')); }));
var ruralMetrics = calculateCoherenceMetrics(
    enrichedCollection.map(function(img) { return img.updateMask(img.select('mask_rural')); }));
var coastalMetrics = calculateCoherenceMetrics(
    enrichedCollection.map(function(img) { return img.updateMask(img.select('mask_coastal')); }));

// ============================================
// CAMADA 6: VISUALIZAÇÃO ARKHE-1
// ============================================

Map.setCenter(-50.0, -15.0, 4); // Brasil central
Map.addLayer(brazil, {color: 'yellow'}, 'Brasil');

// 1. Tendência linear (vermelho = aumento, azul = diminuição)
Map.addLayer(
    urbanMetrics.trend,
    {
      min: 0,
      max: [0.18, 20, -0.18],
      bands: ['scale', 'offset', 'scale'],
      palette: ['blue', 'white', 'red']
    },
    'Tendência Linear (Urbano)',
    true
);

// 2. Coerência Φ (mais verde = mais coerente)
Map.addLayer(
    urbanMetrics.coherence,
    {
      min: 0,
      max: 1,
      palette: ['red', 'yellow', 'green']
    },
    'Coerência Φ (Urbano)',
    true
);

// 3. Sazonalidade (visualização por fase harmônica)
Map.addLayer(
    enrichedCollection.select('harmonic_phase').first(),
    {
      min: -Math.PI,
      max: Math.PI,
      palette: ['blue', 'cyan', 'green', 'yellow', 'red']
    },
    'Fase Harmônica',
    true
);

// ============================================
// CAMADA 7: EXPORTAÇÃO E ANÁLISE ESTATÍSTICA
// ============================================

// Estatísticas zonais por estado brasileiro
var states = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM0_NAME', 'Brazil'));

var zonalStats = urbanMetrics.trend.reduceRegions({
  collection: states,
  reducer: ee.Reducer.mean(),
  scale: 1000
});

// Adicionar camada de polígonos estaduais
Map.addLayer(states, {color: 'white'}, 'Estados', true);

// Exportar resultados
Export.table.toDrive({
  collection: zonalStats,
  description: 'Arkhe1_Brazil_Trend_Analysis',
  folder: 'Arkhe1_Outputs',
  fileFormat: 'CSV'
});

Export.image.toDrive({
  image: urbanMetrics.trend.select('scale').rename('urban_slope'),
  description: 'Arkhe1_Urban_Slope',
  folder: 'Arkhe1_Outputs',
  region: brazil.geometry(),
  scale: 500,
  maxPixels: 1e13
});

// ============================================
// CAMADA 8: VALIDAÇÃO CRUZADA ARKHE-1
// ============================================

print('=== ARKHE-1 GEOGRAPHIC RECOGNITION SYSTEM ===');
print('Coleção DMSP:', dmspCollection.size(), 'imagens');
print('Coleção VIIRS:', viirsCollection.size(), 'imagens');
print('Total unificado:', unifiedCollection.size(), 'imagens');

print('Estatísticas Urbanas (média da inclinação):',
    urbanMetrics.trend.select('scale').reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: brazil.geometry(),
      scale: 1000,
      maxPixels: 1e9
    }));

print('Coerência Média Urbana:',
    urbanMetrics.coherence.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: brazil.geometry(),
      scale: 1000,
      maxPixels: 1e9
    }));

// ============================================
// ANOTAÇÕES ARKHE-1 (LEGADO DO HIPERGRAFO)
// ============================================

/*
 * INTEGRAÇÃO COM O SISTEMA ARKHE-1:
 *
 * 1. CAMADAS DE RECONHECIMENTO:
 *    - L1 (Física): Dados DMSP/VIIRS raw
 *    - L2 (Temporal): Bandas de tempo e fase harmônica
 *    - L3 (Geográfica): Máscaras urbano/rural/costeiro
 *    - L4 (Analítica): Regressões e métricas de coerência Φ
 *    - L5 (Topológica): Relações entre pixels (futuro)
 *
 * 2. CORRESPONDÊNCIA ANYÔNICA:
 *    - A fase harmônica corresponde à estatística α dos ányons
 *    - A coerência Φ é análoga à informação integrada do hipergrafo
 *    - As máscaras geográficas representam diferentes "nós" territoriais
 *
 * 3. PRÓXIMAS ITERAÇÕES:
 *    - Implementar detecção de vórtices (anomalias espaço-temporais)
 *    - Adicionar análise de braiding entre pixels vizinhos
 *    - Integrar com dados de infraestrutura Arkhe-1 em tempo real
 */
