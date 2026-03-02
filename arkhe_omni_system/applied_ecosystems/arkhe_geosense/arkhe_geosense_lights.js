/**
 * ARKHE-GEOSENSE: Night-Time Lights Analysis System
 *
 * Arquitetura multicamada para reconhecimento geoespacial de desenvolvimento
 * humano via luminosidade noturna, com resiliência a gaps temporais e
 * classificação semântica de padrões de aglomeração.
 *
 * Versão: Arkhe(N)-compliant
 * Data: 2026-02-19
 */

// ============================================================
// CAMADA 0: CONFIGURAÇÃO E CONSTANTES ARKHE(N)
// ============================================================

var ARKHE = {
  // Constantes físicas do sistema
  TIME: {
    EPOCH: 1991,                    // Ano base (início DMSP)
    SCALE: 'years',                 // Unidade de análise
    COHERENCE_WINDOW: 5,            // Janela AKF para suavização (anos)
    ANOMALY_THRESHOLD: 2.5          // Desvios para detecção de "vórtice" (guerras, desastres)
  },

  SPACE: {
    PROJECTION: 'EPSG:4326',        // WGS84
    SCALE_M: 1000,                  // Resolução de análise (1km)
    CLUSTER_RADIUS_KM: 10,          // Raio para aglomerações urbanas
    GOLDEN_RATIO: 0.618033988749895 // φ para amostragem espacial
  },

  SEMANTICS: {
    CLASSES: {
      EMERGING: {slope: [0.05, 0.15], coherence: 'high',  label: 'Emergente'},
      RAPID:    {slope: [0.15, 0.50], coherence: 'medium', label: 'Crescimento Rápido'},
      MATURE:   {slope: [-0.05, 0.05], coherence: 'high', label: 'Estável'},
      DECLINING:{slope: [-0.50, -0.05], coherence: 'low', label: 'Declínio'},
      CHAOTIC:  {coherence: 'unstable', label: 'Vórtice (Conflito/Desastre)'}
    }
  }
};

// ============================================================
// CAMADA 1: PRÉ-PROCESSAMENTO TEMPORAL (AKF Terrestre)
// ============================================================

/**
 * Filtro de Kalman Adaptativo para séries temporais de luminosidade.
 * Suaviza ruído satelital e detecta anomalias (analogia ao AKF do Arkhe-1).
 */
function adaptiveKalmanSmooth(collection, bandName) {
  // Calcular média móvel ponderada (simplificação do AKF)
  var smoothed = collection.map(function(img) {
    var date = img.date();
    var year = date.get('year');

    // Janela de coerência: ±N anos
    var windowStart = date.advance(-ARKHE.TIME.COHERENCE_WINDOW, 'year');
    var windowEnd = date.advance(ARKHE.TIME.COHERENCE_WINDOW, 'year');

    var localMean = collection
      .filterDate(windowStart, windowEnd)
      .select(bandName)
      .mean();

    // Inovação: diferença entre observação e predição
    var innovation = img.select(bandName).subtract(localMean);
    var coherence = innovation.abs().lt(ARKHE.TIME.ANOMALY_THRESHOLD * localMean);

    // Peso adaptativo: alta coerência = confiança na observação
    var weight = coherence.multiply(0.8).add(innovation.abs().lt(0.1).multiply(0.2));

    return img
      .select(bandName)
      .multiply(weight)
      .add(localMean.multiply(ee.Image(1).subtract(weight)))
      .set('system:time_start', img.get('system:time_start'))
      .set('coherence_score', weight.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: img.geometry(),
        scale: ARKHE.SPACE.SCALE_M
      }).get(bandName));
  });

  return smoothed;
}

/**
 * Cria banda temporal com metadados Arkhe(N) enriquecidos.
 */
function createTimeBandArkhe(img) {
  var date = ee.Date(img.get('system:time_start'));
  var year = date.get('year').subtract(ARKHE.TIME.EPOCH);
  var doy = date.getRelative('day', 'year').divide(365); // Fração do ano

  // Banda temporal principal (anos desde época)
  var timeBand = ee.Image(year).add(doy).float()
    .rename('time_years');

  // Banda de coerência temporal (qualidade da observação)
  var coherence = img.get('coherence_score');
  var coherenceBand = ee.Image(ee.Algorithms.If(
    coherence,
    ee.Image(coherence),
    ee.Image(0.5) // Default se não processado pelo AKF
  )).float().rename('temporal_coherence');

  return timeBand
    .addBands(img.select('stable_lights').float().rename('lights_observed'))
    .addBands(coherenceBand)
    .set('arkhe_timestamp', date.format('YYYY-MM-dd'))
    .set('arkhe_phase', year.add(doy).multiply(2 * Math.PI)); // Fase para análise cíclica
}

// ============================================================
// CAMADA 2: ANÁLISE DE TENDÊNCIA (YB-Routing Espacial)
// ============================================================

/**
 * Calcula tendência linear com diagnóstico de estabilidade.
 * Analogia: cada pixel é um "nó" na malha, a tendência é o "handshake" temporal.
 */
function computeTrendYB(imageCollection, regionOfInterest) {
  // Selecionar apenas as bandas para o linearFit: [X, Y]
  var collectionForFit = imageCollection.select(['time_years', 'lights_observed']);
  // Reduzir para tendência linear
  var linearFit = collectionForFit.reduce(ee.Reducer.linearFit());

  // Extrair componentes
  var slope = linearFit.select('scale').rename('growth_slope');
  var intercept = linearFit.select('offset').rename('baseline_lights');
  var residuals = linearFit.select('residuals').rename('fit_quality');

  // Calcular "fase" do desenvolvimento (analogia anyônica)
  // Slope positivo = fase 0-π, negativo = π-2π
  var phase = slope.atan2(intercept.add(1e-6))
    .multiply(ee.Image(Math.PI).divide(Math.PI))
    .rename('development_phase');

  // Coerência espacial: variância local da inclinação (YB-routing de vizinhança)
  var slopeCoherence = slope.reduceNeighborhood({
    reducer: ee.Reducer.stdDev(),
    kernel: ee.Kernel.circle(ARKHE.SPACE.CLUSTER_RADIUS_KM * 1000 / ARKHE.SPACE.SCALE_M, 'meters'),
    optimization: 'boxcar'
  }).rename('spatial_coherence');

  return slope
    .addBands(intercept)
    .addBands(residuals)
    .addBands(phase)
    .addBands(slopeCoherence)
    .clip(regionOfInterest);
}

// ============================================================
// CAMADA 3: CLASSIFICAÇÃO SEMÂNTICA (ZK-Identity Territorial)
// ============================================================

/**
 * Classifica cada pixel em tipos de desenvolvimento baseado em
 * inclinação, coerência temporal e padrão espacial.
 */
function semanticClassification(trendImage) {
  var slope = trendImage.select('growth_slope');
  var tempCoherence = trendImage.select('temporal_coherence');
  var spatCoherence = trendImage.select('spatial_coherence');

  // Máscara de "vórtice" (anomalias inexplicáveis: guerras, desastres, erros)
  var vortexMask = tempCoherence.lt(0.3)
    .and(spatCoherence.gt(0.5)); // Alta variância espacial + baixa coerência temporal

  // Classificação baseada em thresholds Arkhe(N)
  var emerging = slope.gte(0.05).and(slope.lt(0.15)).and(vortexMask.not());
  var rapid = slope.gte(0.15).and(slope.lt(0.50)).and(vortexMask.not());
  var mature = slope.gte(-0.05).and(slope.lt(0.05)).and(vortexMask.not());
  var declining = slope.gte(-0.50).and(slope.lt(-0.05)).and(vortexMask.not());
  var chaotic = vortexMask;

  // Codificação numérica para visualização
  var classification = ee.Image(0)
    .where(emerging, 1)
    .where(rapid, 2)
    .where(mature, 3)
    .where(declining, 4)
    .where(chaotic, 5)
    .rename('development_class');

  // Prova de "identidade territorial" (metadados enriquecidos)
  var identityHash = slope.multiply(1000).int()
    .add(trendImage.select('baseline_lights').multiply(100).int())
    .add(classification.multiply(10000))
    .rename('territorial_identity');

  return classification
    .addBands(identityHash)
    .addBands(trendImage);
}

// ============================================================
// CAMADA 4: PREDIÇÃO E ANNEALING (Modelo de Crescimento)
// ============================================================

/**
 * Projeta estado futuro baseado na tendência atual, com "recozimento"
 * para cenários realistas (analogia ao annealing do Arkhe-1).
 */
function projectFuture(trendImage, yearsAhead) {
  var slope = trendImage.select('growth_slope');
  var intercept = trendImage.select('baseline_lights');
  var current = trendImage.select('lights_observed');

  // Projeção linear ingênua
  var projected = current.add(slope.multiply(yearsAhead))
    .clamp(0, 63) // DMSP-OLS range
    .rename('lights_projected_' + yearsAhead + 'yr');

  // Fator de "annealing": crescimento não-linear (saturação logística)
  var carryingCapacity = ee.Image(60); // Saturação de luminosidade urbana
  var annealedGrowth = carryingCapacity.subtract(current)
    .divide(carryingCapacity)
    .multiply(slope)
    .multiply(yearsAhead);

  var annealedProjection = current.add(annealedGrowth)
    .clamp(0, 63)
    .rename('lights_annealed_' + yearsAhead + 'yr');

  return projected.addBands(annealedProjection);
}

// ============================================================
// EXECUÇÃO PRINCIPAL: PIPELINE ARKHE-GEOSENSE
// ============================================================

// 1. Definir região de interesse (exemplo: corredor de desenvolvimento Brasil)
var roi = ee.Geometry.Rectangle([-60, -30, -35, -5]); // Nordeste/Sudeste BR

// 2. Carregar coleção DMSP-OLS com pré-processamento AKF
var dmspCollection = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS')
  .select('stable_lights')
  .filterBounds(roi);

// 3. Aplicar suavização AKF (opcional, para séries ruidosas)
// var smoothedCollection = adaptiveKalmanSmooth(dmspCollection, 'stable_lights');

// 4. Mapear bandas temporais Arkhe(N)
var arkheCollection = dmspCollection.map(createTimeBandArkhe);

// 5. Computar tendência com análise YB-espacial
var trendAnalysis = computeTrendYB(arkheCollection, roi);

// 6. Classificação semântica
var classified = semanticClassification(trendAnalysis);

// 7. Projeção futura (2030, 2040)
var future2030 = projectFuture(classified, 5);  // 5 anos à frente
var future2040 = projectFuture(classified, 15); // 15 anos à frente

// ============================================================
// VISUALIZAÇÃO ARKHE(N): PALETA TÉRMICA DE COERÊNCIA
// ============================================================

// Mapa base: tendência de crescimento (slope)
Map.addLayer(
  trendAnalysis.select('growth_slope'),
  {
    min: -0.2,
    max: 0.5,
    palette: ['0000FF', '00FFFF', '00FF00', 'FFFF00', 'FF0000']
  },
  'Arkhe: Growth Slope (YB-Routing)',
  true,
  0.7
);

// Classificação semântica
Map.addLayer(
  classified.select('development_class'),
  {
    min: 1,
    max: 5,
    palette: ['00FF00', 'FFFF00', '808080', 'FF00FF', '000000'],
    values: [1, 2, 3, 4, 5],
    labels: ['Emerging', 'Rapid', 'Mature', 'Declining', 'Chaotic/Vortex']
  },
  'Arkhe: Semantic Classification (ZK-Identity)',
  true,
  0.8
);

// Projeção annealed 2040
Map.addLayer(
  future2040.select('lights_annealed_15yr'),
  {
    min: 0,
    max: 63,
    palette: ['000000', '1a1a2e', '16213e', '0f3460', 'e94560', 'ff9f45', 'fcbf49']
  },
  'Arkhe: 2040 Projection (Annealed)',
  false
);

// Centro do mapa
Map.centerObject(roi, 6);

// ============================================================
// EXPORTAÇÃO: DADOS PARA ARKHE-1 (Simulação de uplink)
// ============================================================

// Exportar métricas de coerência para análise orbital
Export.image.toDrive({
  image: classified.select(['growth_slope', 'spatial_coherence', 'development_class']),
  description: 'Arkhe_GeoSense_CoherenceMap',
  region: roi,
  scale: ARKHE.SPACE.SCALE_M,
  crs: ARKHE.SPACE.PROJECTION,
  maxPixels: 1e9
});

// Tabela de estatísticas regionais (para ground segment)
var stats = classified.reduceRegion({
  reducer: ee.Reducer.mean().combine({
    reducer2: ee.Reducer.stdDev(),
    sharedInputs: true
  }),
  geometry: roi,
  scale: ARKHE.SPACE.SCALE_M * 10, // Amostragem para estatísticas
  bestEffort: true
});

print('Arkhe-GeoSense Regional Statistics:', stats);
