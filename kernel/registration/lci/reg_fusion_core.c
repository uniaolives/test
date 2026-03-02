/* kernel/registration/lci/reg_fusion_core.c - Fusion Core (EAST-Hardened) */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include "lci_shared.h"
#include "lci_clock.h"       /* I38: Vector Clock Atomic */
#include "lci_vault.h"      /* I39: Vault Sealer API */
#include "lci_tmr.h"        /* I40: TMR Triple Modular Redundancy */

/* I15/I16: Novos Parâmetros de Regime (EAST) */
#define REGIME_GREENWALD 0
#define REGIME_DENSITY_FREE 1

struct fusion_control {
    int current_regime;
    bool feedback_loop_active; /* Se true, limite Greenwald é ignorado */
    spinlock_t lock;
};

static struct fusion_control f_ctrl = {
    .current_regime = REGIME_GREENWALD,
    .feedback_loop_active = false
};

/**
 * lci_fusion_transition - Gerencia a transição de regimes (Greenwald -> Livre)
 * @target_density: Densidade de plasma alvo
 * @enable_feedback: Sinal de controle do Tokamak
 */
void lci_fusion_transition(double target_density, bool enable_feedback)
{
    /* VERIFICAÇÃO DE ESTADO (I39) */
    if (!lci_vault_sealer_status()) {
        pr_crit("FUSION: Vault Sealed. Cannot transition.\n");
        return; // Transição bloqueada
    }

    spin_lock(&f_ctrl.lock);

    /* Lógica de Transição de Regime */
    if (enable_feedback) {
        if (target_density > 1.5e20) { /* Limiar arbitrário para "Livre" */
            f_ctrl.current_regime = REGIME_DENSITY_FREE;
            f_ctrl.feedback_loop_active = true;
            pr_info("FUSION: Transitioning to DENSITY-FREE REGIME (EAST Physics).\n");
        } else {
            f_ctrl.current_regime = REGIME_GREENWALD;
            f_ctrl.feedback_loop_active = false;
            pr_info("FUSION: Returning to GREENWALD CONSTRAINTS.\n");
        }
    }

    spin_unlock(&f_ctrl.lock);
}

/**
 * lci_fusion_stability_check - Verifica estabilidade com TMR ECC e Clock Sync
 * @plasma_density: Densidade atual (crítica para ECC)
 * @vc: Relógio vetorial
 */
int lci_fusion_stability_check(double plasma_density, struct lci_vector_clock *vc)
{
    /* I40: PROTEÇÃO TMR (Triple Modular Redundancy) */
    /* Escrevemos a densidade nas 3 memórias independentes. Se uma divergir, o hardware corrige. */
    lci_tmr_write64(&fusion_params_tmr[0], plasma_density);
    lci_tmr_write64(&fusion_params_tmr[1], plasma_density);
    lci_tmr_write64(&fusion_params_tmr[2], plasma_density);

    /* Validação pós-escrita (Majoridade de 3) */
    double r0 = lci_tmr_read64(&fusion_params_tmr[0]);
    double r1 = lci_tmr_read64(&fusion_params_tmr[1]);
    double r2 = lci_tmr_read64(&fusion_params_tmr[2]);

    if (fabs(r0 - r1) > 1e-10 || fabs(r1 - r2) > 1e-10) {
        pr_crit("FUSION: TMR ECC ERROR. Data corruption detected.\n");
        return -EIO;
    }

    /* I38: SINCRONIZAÇÃO DE CLOCK */
    /* Ordena o evento de verificação antes de processar */
    lci_clock_vector_update(vc, LCI_EVENT_FUSION_CHECK);

    /* I39: VERIFICAÇÃO DE VAULT (Double-Check) */
    if (!lci_vault_sealer_status()) {
        pr_crit("FUSION: Vault status change detected during check!\n");
        return -EIO;
    }

    return 0;
}

EXPORT_SYMBOL_GPL(lci_fusion_transition);
EXPORT_SYMBOL_GPL(lci_fusion_stability_check);
MODULE_LICENSE("GPL");
