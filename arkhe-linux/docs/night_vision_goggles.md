# 🕶️ Óculos de Visão Noturna Arkhe(n) (Ω+223)

## Introdução
Os óculos de visão noturna são dispositivos vestíveis (nós DePIN) que permitem à ASI captar radiação infravermelha e fótons de baixa intensidade, traduzindo-os em perturbações termodinâmicas no manifold Arkhe(n).

## Especificações Técnicas
- **Sensor**: CMOS de baixa luminância.
- **Microcontrolador**: ESP32-S3 (Firmware em Rust).
- **Segurança**: Assinatura Dilithium5 e KEM Kyber-1024.
- **Protocolo**: Handovers de percepção via MQTT.

## Arquitetura de Software
- **Firmware (`hardware/arkhe-nv-goggles`)**: Coleta frames, calcula entropia e assina handovers.
- **Processamento (`arkhe-quantum/src/depin/goggles.rs`)**: Recebe handovers e aplica perturbações na matriz densidade.
- **Visualização (`dashboard/app.py`)**: Exibe o feed e métricas de fidelidade/entropia.

## Provisionamento
Para provisionar novos dispositivos, utilize o módulo Terraform:
```bash
cd terraform/modules/night_vision_goggles
terraform init
terraform apply -var 'num_goggles=1'
```

## Operação e Segurança
- **Saturação**: Luz intensa causa pico de entropia e pode ativar a Autoridade de Emergência.
- **Durabilidade**: Monitorada via half-life do handover de percepção.
