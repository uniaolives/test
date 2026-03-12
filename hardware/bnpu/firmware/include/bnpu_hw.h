// bsp/bnpu_hw.h
#ifndef BNPU_HW_H
#define BNPU_HW_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Endereços de Periféricos (Memory Mapped I/O)
#define TEMP_SENSOR_BASE    0x40005000
#define HEATER_CTRL_BASE    0x40006000
#define PUMP_CTRL_BASE      0x40007000
#define COHERENCE_ADC_BASE  0x40008000

// Estruturas de Controle
typedef struct {
    volatile float current_temp;     // Temperatura atual (Kelvin)
    volatile float target_temp;      // Alvo (ex: 77K, 4.2K, 310K)
    volatile float ramp_rate;        // Taxa de variação (K/s)
} ThermalController;

typedef struct {
    volatile float cpa_conc;         // Concentração do CPA (0.0 a 1.0)
    volatile uint8_t valve_state;    // Estado das válvulas (bitmask)
} PumpController;

typedef struct {
    volatile float lambda_2;         // Coerência local (do sensor Bio-Node)
    volatile bool is_vitrified;      // Estado do vidro
} BioMonitor;

// Funções de Baixo Nível
void hw_thermal_init(void);
void hw_pump_init(void);
void hw_bio_monitor_init(void);

void hw_set_heater_power(float percent);
void hw_set_pump_rate(float ml_min);
void hw_read_sensors(ThermalController* therm, BioMonitor* bio);

// Pressure-Quench Protocol (Hg-1223) HW Interface
void hw_set_anvil_pressure(float gpa);
float hw_read_pressure(void);
void hw_set_cryo_target(float kelvin);
float hw_read_temp(void);

// Stubs for missing functions
float calculate_heater_pid(ThermalController* therm);
float calculate_pump_rate(PumpController* pump);
void delay_ms(uint32_t ms);
uint64_t get_current_epoch(void);
void send_status_update(int state);
void trigger_alarm(const char* msg);

#endif
