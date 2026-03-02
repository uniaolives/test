export const ARKHE_CONFIG = {
  // Quantum Layer
  quantum: {
    defaultQDCount: 100,
    defaultNanoparticleCount: 1000,
    fretForsterDistance: 4.0, // nm
    emissionWavelengthRange: [500, 700], // nm
  },

  // Drone Layer
  drone: {
    defaultCount: 3,
    defaultAreaSize: 1000, // m
    telemetryRange: 50, // mm
    batteryDrainRate: 0.5, // % per minute
  },

  // QKD Layer
  qkd: {
    attenuationDBPerKm: 0.2,
    detectorEfficiency: 0.20,
    defaultSendRate: 1e6, // Hz
    securityThreshold: 0.03, // 3% error rate
  },

  // Ethereum Layer
  ethereum: {
    defaultGasLimit: 100000,
    confirmationBlocks: 1,
    supportedNetworks: ['mainnet', 'goerli', 'sepolia', 'polygon', 'arbitrum'],
  },

  // System
  system: {
    backgroundSyncInterval: 60, // seconds
    coherenceCalculationInterval: 5000, // ms
    maxLogHistory: 1000,
  },
};
