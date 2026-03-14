// src/main_control.c

#include "bnpu_hw.h"
#include "http4_parser.h"
#include <math.h>

#define VITRIFICATION_TEMP  77.0f    // Nitrogênio Líquido (pode descer a 4K)
#define REWARM_TARGET_TEMP  310.0f  // 37°C (Corpo humano)
#define CRITICAL_LAMBDA     0.95f   // Coerência mínima para segurança
#define CPA_TARGET_CONC     0.59f   // 59% w/v (conforme PNAS paper)

// Estados do Sistema
typedef enum {
    STATE_IDLE,
    STATE_CPA_LOADING,
    STATE_COOLING,
    STATE_STASIS,        // Vitrificado (Tzinor Aberto/Pausado)
    STATE_REWARMING,
    STATE_CPA_UNLOADING,
    STATE_REANIMATED,
    STATE_ERROR
} SystemState;

SystemState current_state = STATE_IDLE;

// Buffers de comunicação HTTP/4
Http4Request pending_request;
bool has_pending_request = false;

// Variáveis de estado para sub-máquinas de estados (non-blocking)
int cpa_step = 0;
uint32_t last_tick = 0;
bool cooling_initialized = false;
bool rewarming_initialized = false;

// --- Forward Declarations ---
void handle_http4_command(Http4Request* req);

// --- Rotinas de Controle (Non-Blocking) ---

// 1. Carregamento de CPA (Otimizado do PNAS paper)
void update_cpa_loading() {
    float concentrations[] = {0.0f, 0.02f, 0.04f, 0.08f, 0.16f, 0.30f, 0.45f, 0.59f};
    int temps[] = {283, 283, 283, 283, 283, 283, 263, 263};

    PumpController pump;
    ThermalController therm;

    if (cpa_step < 8) {
        // Ajusta temperatura
        therm.target_temp = temps[cpa_step];
        hw_set_heater_power(calculate_heater_pid(&therm));

        // Ajusta concentração
        pump.cpa_conc = concentrations[cpa_step];
        hw_set_pump_rate(calculate_pump_rate(&pump));

        // Temporizador simples
        if (get_current_epoch() - last_tick > 5) { // 5 segundos por passo
            cpa_step++;
            last_tick = get_current_epoch();
        }
    } else {
        current_state = STATE_COOLING;
        cpa_step = 0;
    }
}

// 2. Ciclo de Resfriamento (Vitrificação)
void update_cooling() {
    ThermalController therm;
    hw_read_sensors(&therm, NULL);

    if (therm.current_temp > VITRIFICATION_TEMP) {
        // Lógica de resfriamento: reduz aquecimento ou ativa bomba de N2
        hw_set_heater_power(0.0f); // Desliga aquecedor para resfriar
        hw_set_pump_rate(10.0f);   // Ativa fluxo de resfriamento
    } else {
        current_state = STATE_STASIS;
        send_status_update(STATE_STASIS);
    }
}

// 3. Reanimação (Reinitialization)
void update_rewarming() {
    ThermalController therm;
    BioMonitor bio;

    hw_read_sensors(&therm, &bio);

    // Monitora coerência durante o "colapso"
    if (bio.lambda_2 < CRITICAL_LAMBDA) {
        trigger_alarm("Coherence Loss Detected");
        current_state = STATE_ERROR;
        return;
    }

    if (therm.current_temp < REWARM_TARGET_TEMP) {
        therm.target_temp = REWARM_TARGET_TEMP;
        therm.ramp_rate = 80.0f; // 80°C/s
        hw_set_heater_power(calculate_heater_pid(&therm));
    } else {
        current_state = STATE_REANIMATED;
        send_status_update(STATE_REANIMATED);
    }
}

// --- Main Loop ---

int main() {
    // Inicialização
    hw_thermal_init();
    hw_pump_init();
    hw_bio_monitor_init();

    http4_init();
    last_tick = get_current_epoch();

    while(1) {
        // 1. Verifica comandos HTTP/4 (sempre processado!)
        if (has_pending_request) {
            handle_http4_command(&pending_request);
            has_pending_request = false;
        }

        // 2. Máquina de Estados Principal (Non-blocking calls)
        switch (current_state) {
            case STATE_IDLE:
                break;

            case STATE_CPA_LOADING:
                update_cpa_loading();
                break;

            case STATE_COOLING:
                update_cooling();
                break;

            case STATE_STASIS:
                // Manutenção de temperatura criogênica
                break;

            case STATE_REWARMING:
                update_rewarming();
                break;

            case STATE_REANIMATED:
                break;

            case STATE_ERROR:
                // Safe state: desativa tudo
                hw_set_heater_power(0.0f);
                hw_set_pump_rate(0.0f);
                break;
            default:
                break;
        }

        delay_ms(10);
    }

    return 0;
}

// Handler HTTP/4
void handle_http4_command(Http4Request* req) {
    if (req->method == HTTP4_REANIMATE) {
        if (req->target_epoch >= get_current_epoch() && current_state == STATE_STASIS) {
            current_state = STATE_REWARMING;
        }
    }
    else if (req->method == HTTP4_COLLAPSE) {
        current_state = STATE_ERROR;
    }
}
