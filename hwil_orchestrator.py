#!/usr/bin/env python3
"""
Arkhe(N) 24-Hour HWIL Orchestrator
Inicia, monitora e consolida métricas da bancada de teste termodinâmica.
"""

import subprocess
import time
import sys
import datetime
import csv
import os

# Configurações do Teste
TEST_DURATION_HOURS = 24
TEST_DURATION_SECONDS = TEST_DURATION_HOURS * 3600
PSI_THRESHOLD = 0.847

def print_log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] 🛰️ {msg}")

def run_24h_test():
    print_log(f"Iniciando Maratona HWIL Arkhe(N) de {TEST_DURATION_HOURS} horas...")

    # 1. Iniciar Atmosfera GNU Radio (Entropia)
    print_log("Injetando Entropia (GNU Radio Atmosphere)...")
    atmosphere_proc = None
    if os.path.exists("arkhen_atmosphere.py"):
        atmosphere_proc = subprocess.Popen(
            ["python3", "arkhen_atmosphere.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        print_log("Aviso: arkhen_atmosphere.py não encontrado. Pulando etapa 1.")

    time.sleep(5) # Aguarda os blocos de RF estabilizarem

    # 2. Iniciar Nó Rust (O Transdutor e Consenso)
    print_log("Iniciando Transdutor Arkhe(N) (Rust Core)...")
    # Assumindo que o binário está no workspace e pode ser executado
    rust_proc = subprocess.Popen(
        ["cargo", "run", "--release"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    start_time = time.time()
    log_file = open("arkhen_24h_telemetry.csv", "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Timestamp", "Uptime(s)", "HandshakeStatus", "GlobalCoherence", "PhaseError"])

    anomalies = 0

    try:
        while True:
            current_time = time.time()
            uptime = current_time - start_time

            if uptime >= TEST_DURATION_SECONDS:
                print_log("✅ Tempo de teste concluído com sucesso!")
                break

            # Lendo a saída do Rust em tempo real
            line = rust_proc.stdout.readline()
            if not line:
                break

            line = line.strip()

            # Parsing da telemetria Rust
            if "Handshake aceito" in line or "Handshake pendente" in line:
                status = "ACCEPTED" if "aceito" in line else "PENDING"

                # Extrai a coerência do log, ex: "Coerência global: 0.965"
                coherence = 0.0
                if "Coerência global:" in line:
                    parts = line.split("Coerência global: ")
                    if len(parts) > 1:
                        try:
                            coherence = float(parts[1].split()[0][:5])
                        except:
                            pass

                # Extrai erro de fase se presente
                phase_error = 0.0
                if "Phase error:" in line:
                     parts = line.split("Phase error: ")
                     if len(parts) > 1:
                        try:
                            phase_error = float(parts[1].split()[0])
                        except:
                            pass

                csv_writer.writerow([datetime.datetime.now().isoformat(), round(uptime, 2), status, coherence, phase_error])
                log_file.flush()

                # Atuação do SafeCore local
                if coherence > 0 and coherence < PSI_THRESHOLD:
                    print_log(f"⚠️ ALERTA DE DECOERÊNCIA: {coherence} caiu abaixo do limite Ψ ({PSI_THRESHOLD})!")
                    anomalies += 1

            # Status a cada hora
            if int(uptime) > 0 and int(uptime) % 3600 == 0:
                print_log(f"Progresso: {int(uptime)/3600:.0f}/{TEST_DURATION_HOURS} horas concluídas. Anomalias: {anomalies}")

    except KeyboardInterrupt:
        print_log("Teste interrompido manualmente (Ctrl+C).")

    finally:
        # Encerramento Gracioso
        print_log("Desligando hardware e consolidando matrizes...")
        rust_proc.terminate()
        if atmosphere_proc:
            atmosphere_proc.terminate()
        log_file.close()
        rust_proc.wait()
        if atmosphere_proc:
            atmosphere_proc.wait()

        print_log(f"Relatório Final: Teste encerrou com {anomalies} quedas abaixo do limite ético Ψ.")

if __name__ == "__main__":
    run_24h_test()
