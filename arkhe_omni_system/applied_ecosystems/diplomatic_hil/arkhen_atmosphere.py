#!/usr/bin/env python3
"""
Arkhe(N) Entropic Atmosphere Simulator
Injeta Doppler dinâmico e AWGN para testes HWIL entre LEO e MEO.
"""

from gnuradio import gr, blocks, analog, channels
import time
import math
import threading

class ArkhenAtmosphere(gr.top_block):
    def __init__(self, sample_rate=2e6, carrier_freq=437e6):
        super(ArkhenAtmosphere, self).__init__("Arkhen_Doppler_Channel")
        self.samp_rate = sample_rate
        self.carrier_freq = carrier_freq

        # 1. Fonte do Sinal (SDR TX via Loopback ou ZMQ)
        self.source = blocks.null_source(gr.sizeof_gr_complex) # Placeholder para o TX real

        # 2. Modelo de Canal Dinâmico (Injeta AWGN e Fading)
        self.channel_model = channels.dynamic_channel_model(
            samp_rate=self.samp_rate,
            sro_stddev=0.0,
            sro_max_dev=0.0,
            cfo_stddev=0.0,
            cfo_max_dev=0.0,
            N=8,
            fD=0.0, # Frequência Doppler inicial
            LOS=True,
            K=10.0, # Fator Rician (Linha de visada forte no espaço)
            seed=0,
            noise_amp=0.05 # AWGN que degrada o C_local
        )

        # 3. Multiplicador de Frequência para Shift Doppler Exato
        self.doppler_rotator = blocks.rotator_cc(0)

        # 4. Saída do Sinal (SDR RX)
        self.sink = blocks.null_sink(gr.sizeof_gr_complex) # Placeholder para o RX real

        # Conexões do Grafo
        self.connect((self.source, 0), (self.channel_model, 0))
        self.connect((self.channel_model, 0), (self.doppler_rotator, 0))
        self.connect((self.doppler_rotator, 0), (self.sink, 0))

    def set_doppler(self, shift_hz):
        """Atualiza a rotação de fase para simular o Doppler shift"""
        phase_increment = (2 * math.pi * shift_hz) / self.samp_rate
        self.doppler_rotator.set_phase_inc(phase_increment)

def orbital_entropy_injector(tb):
    """Calcula a física orbital e injeta a entropia térmica/cinética no canal"""
    c = 299792458.0 # m/s
    v_leo = 7600.0  # Starlink
    v_meo = 3900.0  # Galileo

    print("🌌 [ATMOSFERA] Iniciando injeção de entropia orbital...")
    time_s = 0

    while True:
        # Simulando uma passagem orbital (aproximação e afastamento)
        # Velocidade relativa varia de V_max para -V_max
        v_rel = (v_leo - v_meo) * math.cos(time_s / 20.0)

        # Cálculo do Efeito Doppler exato: fd = (v/c) * fc
        doppler_shift = (v_rel / c) * tb.carrier_freq

        tb.set_doppler(doppler_shift)

        if int(time_s) % 5 == 0:
            print(f"⏱️ t={time_s:03}s | V_rel: {v_rel:7.1f} m/s | Doppler Shift: {doppler_shift:7.1f} Hz")

        time.sleep(0.1)
        time_s += 0.1

if __name__ == '__main__':
    tb = ArkhenAtmosphere()
    tb.start()

    # Inicia a thread que destrói a fase ideal do sinal
    entropy_thread = threading.Thread(target=orbital_entropy_injector, args=(tb,))
    entropy_thread.daemon = True
    entropy_thread.start()

    try:
        print("Pressione Ctrl+C para encerrar a simulação atmosférica...")
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    tb.stop()
    tb.wait()
