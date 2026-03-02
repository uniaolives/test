/* tests/tim_stress_trigger.c
 * Módulo de kernel temporário para DOS (Denial of Service) no próprio validador
 * Objetivo: Validar estabilidade sob carga extrema.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include "../kernel/registration/reg_shared.h"

// Importa função do módulo principal (precisa exportar symbol)
extern int reg_validator_request_check(struct task_struct *tsk, float *hidden,
                                       int n, int dim, float acc);

static struct task_struct *stress_thread;
static bool keep_running = true;

static int stress_loop(void *data) {
    float dummy_states[64]; // Mock buffer
    int count = 0;

    pr_info("TIM-STRESS: Starting high-frequency injection...\n");

    while (!kthread_should_stop() && keep_running) {
        // Simula um PID e dados aleatórios
        struct task_struct *mock_task = current; // Usa o próprio thread

        // Envia pedido
        int ret = reg_validator_request_check(mock_task, dummy_states,
                                            10, 64, 0.99f);

        if (ret < 0) {
            pr_err("TIM-STRESS: Failed at iteration %d (buffer full?)\n", count);
            msleep(10); // Backoff
        }

        count++;
        if (count % 1000 == 0) {
            pr_info("TIM-STRESS: Injecting... %d reqs sent\n", count);
            msleep(100); // Breve pausa a cada 1k
        }

        // Frequência alvo: ~10k Hz (sem sleep no loop apertado)
        cond_resched();
    }
    return 0;
}

static int __init stress_init(void) {
    stress_thread = kthread_run(stress_loop, NULL, "tim_stress_d");
    return 0;
}

static void __exit stress_exit(void) {
    keep_running = false;
    if (stress_thread) {
        kthread_stop(stress_thread);
    }
    pr_info("TIM-STRESS: Stopped.\n");
}

module_init(stress_init);
module_exit(stress_exit);
MODULE_LICENSE("GPL");
