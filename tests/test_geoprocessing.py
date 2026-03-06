#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TEST: GEOPROCESSING ENGINE & SPATIAL QUALITY CONTROL

import numpy as np
import pytest
import sys
import os

# Ajustar o path para encontrar o engine
sys.path.append(os.path.join(os.getcwd(), 'arkhe_omni_system/applied_ecosystems/arkhe_geosense'))
from geoprocessing_engine import GeoprocessingEngine

def test_clipping_logic():
    engine = GeoprocessingEngine()
    data = np.ones((10, 10))
    mask = np.zeros((10, 10))
    mask[0:5, 0:5] = 1 # Recortar quadrante superior esquerdo

    result = engine.perform_clipping(data, mask)

    assert result.shape == (10, 10)
    assert np.sum(result) == 25
    assert result[0, 0] == 1
    assert result[6, 6] == 0

def test_thematic_interpretation():
    engine = GeoprocessingEngine()
    # Imagem simulada com diferentes brilhos (albedo)
    # 0.05 = Água, 0.4 = Parques, 0.9 = Ferrovias
    img = np.array([[0.05, 0.4], [0.9, 0.2]])

    layers = engine.interpret_thematic_layers(img)

    assert layers['hydro_poly'][0, 0] == 1
    assert layers['parks_plazas'][0, 1] == 1
    assert layers['railways'][1, 0] == 1
    assert layers['railways'][1, 1] == 0

def test_quality_control():
    engine = GeoprocessingEngine(resolution=10.0)
    layer_data = np.random.randint(0, 2, (100, 100))

    report = engine.validate_quality("Test_Layer", layer_data)

    assert report['layer'] == "Test_Layer"
    assert report['status'] == "PASSED"
    assert report['resolution'] == 10.0
    assert 0 <= report['feature_density'] <= 1

def test_resampling_resolution_update():
    engine = GeoprocessingEngine(resolution=30.0)
    data = np.random.rand(60, 60)

    # Reamostrar para 15m (deve dobrar as dimensões no protótipo)
    resampled = engine.resample_data(data, 15.0)

    assert engine.resolution == 15.0
    assert resampled.shape == (120, 120)

if __name__ == "__main__":
    # Execução manual simplificada
    print("[*] Rodando testes de geoprocessamento...")
    test_clipping_logic()
    test_thematic_interpretation()
    test_quality_control()
    test_resampling_resolution_update()
    print("✅ Todos os testes passaram!")
