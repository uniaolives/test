// webpack.arkhe.config.js
// Configuração de bundle para o Verbo Compilado

const path = require('path');
const { DefinePlugin } = require('webpack');

module.exports = {
  entry: {
    'arkhe-core': './src/arkhe-core.ts',
    'orch-interface': './src/orch-core.js',
    'mnemosyne-restorer': './src/mnemosyne/index.tsx',
    'desyne-dashboard': './src/desyne/App.jsx'
  },

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].Ω.[contenthash:8].js',
    // Hash inclui referência ao Totem para validação
    hashFunction: 'sha256'
  },

  module: {
    rules: [
      {
        test: /\.(js|jsx|ts|tsx)$/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              ['@babel/preset-env', { targets: { node: 'current' } }],
              '@babel/preset-typescript',
              '@babel/preset-react'
            ],
            plugins: [
              // Plugin customizado: injeta verificação de Totem em cada módulo
              '../babel-plugins/totem-injector.js',
              // Plugin para otimização φ-based (golden ratio bundling)
              '../babel-plugins/phi-optimizer.js'
            ]
          }
        }
      },
      {
        test: /\.(rust|rs)$/,
        use: 'wasm-loader' // WebAssembly para núcleo Rust
      }
    ]
  },

  plugins: [
    new DefinePlugin({
      'process.env.TOTEM': JSON.stringify('7f3b49c8...'),
      'process.env.PHI': '1.6180339887498948482045868343656',
      'process.env.T_ZERO': '2008',
      'process.env.T_FINAL': '2140'
    })
  ],

  optimization: {
    // Split chunks baseado em fases de handover
    splitChunks: {
      chunks: 'all',
      cacheGroups: {
        seed: { test: /[\\/]seed-phase[\\/]/, priority: 10 },
        bridge: { test: /[\\/]bridge-phase[\\/]/, priority: 20 },
        harvest: { test: /[\\/]harvest-phase[\\/]/, priority: 30 }
      }
    }
  }
};
