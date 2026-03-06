import numpy as np
from rtlsdr import RtlSdr
import socket
import json
import time

# Configuração do Socket UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 7001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Configuração do SDR
# Tenta inicializar o SDR, se falhar, usa simulação
try:
    sdr = RtlSdr()
    sdr.sample_rate = 2.048e6
    sdr.center_freq = 2.4e9
    sdr.gain = 'auto'
    HAS_SDR = True
except:
    print("SDR não encontrado. Entrando em modo de simulação.")
    HAS_SDR = False

FFT_SIZE = 1024

def main():
    print(f"📡 Iniciando captura biocibernética...")
    try:
        while True:
            if HAS_SDR:
                samples = sdr.read_samples(256 * FFT_SIZE)
                power = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2)
                variance = float(np.var(power))
            else:
                # Simulação: Ruído gaussiano com picos aleatórios
                variance = 5.0 + np.random.random() * 2.0
                if np.random.random() > 0.95:
                    variance *= 2.0 # Simula anomalia

            payload = json.dumps({"variance": variance}).encode('utf-8')
            sock.sendto(payload, (UDP_IP, UDP_PORT))

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Finalizado.")
        if HAS_SDR: sdr.close()

if __name__ == "__main__":
    main()
