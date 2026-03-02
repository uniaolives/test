//! Arkhe(n) DSP Module — Physical Layer Interface
//! Conecta o protocolo Diplomático à realidade eletromagnética via SDR

use num_complex::Complex32;
use std::f32::consts::PI;

// Mocking SoapySDR if not available
pub mod soapysdr_mock {
    use num_complex::Complex32;
    #[derive(Debug)]
    pub enum Error {
        MockError,
    }
    pub struct Device;
    impl Device {
        pub fn new(_args: &str) -> Result<Self, Error> { Ok(Device) }
        pub fn set_sample_rate(&self, _dir: Direction, _ch: usize, _rate: f64) -> Result<(), Error> { Ok(()) }
        pub fn set_frequency(&self, _dir: Direction, _ch: usize, _freq: f64, _args: ()) -> Result<(), Error> { Ok(()) }
        pub fn set_gain(&self, _dir: Direction, _ch: usize, _gain: f64) -> Result<(), Error> { Ok(()) }
        pub fn rx_stream<T>(&self, _channels: &[usize]) -> Result<RxStream<T>, Error> { Ok(RxStream(std::marker::PhantomData)) }
    }
    pub enum Direction { Rx }
    pub struct RxStream<T>(std::marker::PhantomData<T>);
    impl RxStream<Complex32> {
        pub fn activate(&self, _time: Option<i64>) -> Result<(), Error> { Ok(()) }
        pub fn read(&self, buffers: &mut [&mut [Complex32]], _timeout: i64) -> Result<usize, Error> {
            // Fill with some mock data (high power samples for coherence)
            if !buffers.is_empty() && !buffers[0].is_empty() {
                for i in 0..buffers[0].len() {
                    buffers[0][i] = Complex32::new(10.0, 0.0);
                }
                return Ok(buffers[0].len());
            }
            Ok(0)
        }
    }
}

// Use real soapysdr or mock
#[cfg(feature = "real_sdr")]
use soapysdr::{Device, RxStream, Direction};
#[cfg(not(feature = "real_sdr"))]
use soapysdr_mock::{Device, RxStream, Direction};

#[cfg(not(feature = "real_sdr"))]
pub type SoapyError = soapysdr_mock::Error;
#[cfg(feature = "real_sdr")]
pub type SoapyError = soapysdr::Error;

/// Configurações do SDR e do Loop de Fase
const SAMPLE_RATE: f64 = 2e6; // 2 MSPS
const FREQUENCY: f64 = 437e6; // UHF Band
const PLL_BANDWIDTH: f32 = 0.01; // Ajuste agressivo para rastrear Doppler rápido

pub struct HardwareEmbassy {
    pub device: Device,
    pub rx_stream: RxStream<Complex32>,
    pub phase_estimate: f32, // Fase rastreada do nó remoto
    pub freq_estimate: f32,  // Estimativa de Doppler residual
}

impl HardwareEmbassy {
    /// Inicializa a conexão com o rádio (Ex: HackRF, USRP ou RTL-SDR)
    pub fn new(device_args: &str) -> Result<Self, SoapyError> {
        let device = Device::new(device_args)?;
        device.set_sample_rate(Direction::Rx, 0, SAMPLE_RATE)?;
        device.set_frequency(Direction::Rx, 0, FREQUENCY, ())?;
        device.set_gain(Direction::Rx, 0, 40.0)?; // AGC Manual

        let rx_stream = device.rx_stream::<Complex32>(&[0])?;

        Ok(Self {
            device,
            rx_stream,
            phase_estimate: 0.0,
            freq_estimate: 0.0,
        })
    }

    /// Ativa o stream de recepção
    pub fn activate(&mut self) -> Result<(), SoapyError> {
        self.rx_stream.activate(None)?;
        Ok(())
    }

    /// Processa um bloco de amostras I/Q, executa o PLL e extrai a Coerência e a Fase
    pub fn extract_phase_and_coherence(&mut self) -> Result<(f64, f64), SoapyError> {
        let mut buffer = vec![Complex32::new(0.0, 0.0); 1024];
        let num_read = self.rx_stream.read(&mut [&mut buffer], 1000000)?;

        let mut signal_power = 0.0;
        let mut phase_sum = 0.0;

        // PLL Digital (Costas Loop de 1ª ordem simplificado) para rastreamento de fase
        for i in 0..num_read {
            let sample = buffer[i];

            // Potência do sinal = I^2 + Q^2 (proxy para C_local)
            let power = sample.norm_sqr();
            signal_power += power;

            // Extração de Fase Bruta: atan2(Q, I)
            let raw_phase = sample.im.atan2(sample.re);

            // Cálculo do erro de fase
            let mut phase_error = raw_phase - self.phase_estimate;

            // Normalizar erro para o intervalo [-π, π]
            while phase_error > PI { phase_error -= 2.0 * PI; }
            while phase_error < -PI { phase_error += 2.0 * PI; }

            // Atualização do PLL (Rastreando o Doppler)
            self.freq_estimate += PLL_BANDWIDTH * PLL_BANDWIDTH * 0.25 * phase_error;
            self.phase_estimate += PLL_BANDWIDTH * phase_error + self.freq_estimate;

            // Normalizar a estimativa da fase
            while self.phase_estimate > PI { self.phase_estimate -= 2.0 * PI; }
            while self.phase_estimate < -PI { self.phase_estimate += 2.0 * PI; }

            phase_sum += self.phase_estimate;
        }

        // Média da fase no bloco atual
        let avg_phase = (phase_sum / num_read as f32) as f64;

        // Coerência C baseada na relação Sinal-Ruído (SNR) e travamento do PLL
        let avg_power = (signal_power / num_read as f32) as f64;
        let coherence = (1.0 - (1.0 / (1.0 + avg_power))).min(0.99); // Normalização simplificada

        Ok((avg_phase, coherence))
    }
}
