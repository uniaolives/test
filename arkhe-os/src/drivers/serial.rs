//! Serial Communication Protocols (CAN, SPI, I2C, RS-485)
//! Simulated implementation for the Arkhe(n) Retrocausality Package.

use rand::Rng;

#[derive(Debug, Clone)]
pub enum SerialProtocol {
    CAN,
    SPI,
    I2C,
    RS485,
    RS232,
}

pub struct SerialFrame {
    pub protocol: SerialProtocol,
    pub identifier: u32,
    pub data: Vec<u8>,
}

pub struct SerialController {
    pub baud_rate: u32,
}

impl SerialController {
    pub fn new(baud_rate: u32) -> Self {
        Self { baud_rate }
    }

    /// Read a CAN bus frame
    pub fn read_can_frame(&self) -> SerialFrame {
        let mut rng = rand::thread_rng();
        let mut data = vec![0u8; 8];
        rng.fill(&mut data[..]);

        SerialFrame {
            protocol: SerialProtocol::CAN,
            identifier: rng.gen_range(0..0x7FF),
            data,
        }
    }

    /// Perform an SPI transfer
    pub fn spi_transfer(&self, tx_data: &[u8]) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let mut rx_data = vec![0u8; tx_data.len()];
        rng.fill(&mut rx_data[..]);
        rx_data
    }

    /// Read from an I2C device
    pub fn i2c_read(&self, address: u8, length: usize) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let mut data = vec![0u8; length];
        rng.fill(&mut data[..]);
        data
    }

    /// Send data via RS-485 (Differential Serial)
    pub fn rs485_send(&self, data: &[u8]) {
        println!("[RS-485] Sending {} bytes at {} bps", data.len(), self.baud_rate);
    }
}
