/* kernel/registration/lci/reg_cfm_sleep.c - CFM Resonance Integrator (2t20)
 * INVARIANTES: I38 (Clock Sync), I39 (Vault Sealer), I40 (TMR ECC)
 * Mapeia attractors φ/ψ do Constrained Field Model para manifold LCI
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/atomic.h>
#include "lci_shared.h"
#include "lci_clock.h"
#include "lci_vault.h"
#include "lci_tmr.h"
#include "cfm_dynamics.h"

/* I40: Shadow State TMR para parâmetros CFM (triplicação automática) */
static struct lci_tmr_state cfm_phi_tmr;    /* Parâmetro φ (ordem de campo) */
static struct lci_tmr_state cfm_psi_tmr;    /* Parâmetro ψ (deformação) */
static struct lci_tmr_state cfm_stability_tmr; /* S = 0.94 (fase nematic) */

/* I38: Clock Vector dedicado para eventos de sono/ressonância CFM */
static struct lci_vector_clock cfm_sleep_vc;

/* Mutex para transições atômicas de regime CFM */
static DEFINE_MUTEX(cfm_sleep_mutex);

/* Constantes de normalização para alvo 2t20 */
#define CFM_PHI_NORMALIZATION 1.02
#define CFM_PSI_NORMALIZATION 0.51

/**
 * cfm_execute_resonance_tuning - Aplica alvo 2t20 com garantias I38-I40
 * @text_mode: Payload do comando (ex: 0xA1)
 * @target_value: Valor 2t20 (ressonância alvo)
 * @vc: Clock vector para ordenamento causal (I38)
 */
int cfm_execute_resonance_tuning(int text_mode, double target_value,
                                 struct lci_vector_clock *vc)
{
    int ret = 0;
    double phi_new, psi_new;

    /* I38: Ordenar início da transação de ressonância */
    lci_clock_vector_update(vc, LCI_EVENT_RESONANCE_TUNING_START);

    pr_info("CFM-SLEEP: Alvo [text=0x%x, 2t20=%.4f] recebido\n", text_mode, target_value);

    /* I39: Verificar Vault antes de qualquer mudança de estado */
    if (lci_vault_sealer_status() == VAULT_SEALED) {
        pr_crit("CFM-SLEEP: Vault SEALED. Ressonância BLOQUEADA (I39).\n");
        return -EACCES;
    }

    /* I39.1: Iniciar transação atômica */
    ret = lci_vault_begin_transaction();
    if (ret != 0) {
        pr_err("CFM-SLEEP: Não pode iniciar transação (I39): %d\n", ret);
        return -EBUSY;
    }

    /* I40: Ler estado atual (majoridade 2/3) */
    double phi_current = lci_tmr_read64(&cfm_phi_tmr);
    double psi_current = lci_tmr_read64(&cfm_psi_tmr);

    if (isnan(phi_current) || isnan(psi_current)) {
        pr_emerg("CFM-SLEEP: TMR ECC corruption detectado! Abortando (I40).\n");
        ret = -EIO;
        goto abort_transaction;
    }

    /* Aplicar alvo 2t20: φ_new = target * norm, ψ_new = target * norm */
    phi_new = target_value * CFM_PHI_NORMALIZATION;
    psi_new = target_value * CFM_PSI_NORMALIZATION;

    /* I40.1: Escrever NOVO estado com proteção TMR (triplicação automática) */
    /* ATENÇÃO: Escrever CADA parâmetro individualmente via TMR */
    lci_tmr_write64(&cfm_phi_tmr, phi_new);
    lci_tmr_write64(&cfm_psi_tmr, psi_new);

    /* I38: Tick após atualização de estado (marca commit lógico) */
    lci_clock_vector_tick(vc);

    pr_info("CFM-SLEEP: Ressonância sintonizada. φ: %.2f→%.2f, ψ: %.2f→%.2f (I40)\n",
            phi_current, phi_new, psi_current, psi_new);

    /* I39.2: Commit atômico */
    ret = lci_vault_commit();
    if (ret != 0) {
        pr_emerg("CFM-SLEEP: Vault commit FALHOU! Rollback (I39).\n");
        /* Rollback: Restaurar valores anteriores */
        lci_tmr_write64(&cfm_phi_tmr, phi_current);
        lci_tmr_write64(&cfm_psi_tmr, psi_current);
        ret = -EIO;
        goto abort_transaction;
    }

    return 0;

abort_transaction:
    lci_vault_rollback();
    /* I38: Ordenar rollback */
    lci_clock_vector_update(vc, LCI_EVENT_RESONANCE_ROLLBACK);
    lci_clock_vector_tick(vc);
    return ret;
}

/**
 * cfm_resonance_step - Passo de evolução do attractor (I38-I40)
 * @phi_psi_state: Estado φ/ψ (será atualizado)
 * @vc: Clock vector para ordering
 */
static int cfm_resonance_step(double *phi_psi_state, struct lci_vector_clock *vc)
{
    double phi_protected, psi_protected, phi_evolved;
    int ret = 0;

    /* I38: Ordenar passo de ressonância */
    lci_clock_vector_update(vc, LCI_EVENT_RESONANCE_STEP);

    /* I39: Verificar Vault */
    if (lci_vault_sealer_status() == VAULT_SEALED) {
        pr_warn("CFM-SLEEP: Vault SEALED. Passo BLOQUEADO (I39).\n");
        return -EACCES;
    }

    /* I39.1: Transação atômica */
    ret = lci_vault_begin_transaction();
    if (ret != 0) return -EBUSY;

    /* I40: Ler estado protegido */
    phi_protected = lci_tmr_read64(&cfm_phi_tmr);
    psi_protected = lci_tmr_read64(&cfm_psi_tmr);

    if (isnan(phi_protected) || isnan(psi_protected)) {
        pr_emerg("CFM-SLEEP: TMR corruption! Abortando.\n");
        ret = -EIO;
        goto abort_transaction;
    }

    /* CFM dynamics: evoluir estado */
    /* IMPORTANTE: CFM espera array [phi, psi], passar valores individuais */
    double state_array[2] = { phi_protected, psi_protected };
    phi_evolved = cfm_step(state_array, 0.94);  // S=0.94 nematic

    /* I40.1: Escrever estado evoluído */
    lci_tmr_write64(&cfm_phi_tmr, phi_evolved);

    /* I38: Tick final */
    lci_clock_vector_tick(vc);

    /* Retornar valor atualizado */
    *phi_psi_state = phi_evolved;

    /* I39.2: Commit */
    ret = lci_vault_commit();
    if (ret != 0) {
        pr_emerg("CFM-SLEEP: Commit falhou! Rollback.\n");
        *phi_psi_state = phi_protected;  // Restaurar valor seguro
        ret = -EIO;
        goto abort_transaction;
    }

    pr_debug("CFM-SLEEP: Passo evoluído φ: %.6f (I38-I40)\n", phi_evolved);
    return 0;

abort_transaction:
    lci_vault_rollback();
    lci_clock_vector_update(vc, LCI_EVENT_RESONANCE_ROLLBACK);
    lci_clock_vector_tick(vc);
    return ret;
}

/* Handler de ioctl para comando user-space (ex: echo 2t20 > /dev/cfm) */
static long cfm_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    struct cfm_tune_params params;
    int ret;

    if (cmd != CFM_TUNE_RESONANCE) return -EINVAL;

    if (copy_from_user(&params, (void __user *)arg, sizeof(params)))
        return -EFAULT;

    /* I38: Passar clock vector global para ordenamento */
    ret = cfm_execute_resonance_tuning(params.text_mode, params.target_value,
                                       &cfm_sleep_vc);
    return ret;
}

/* Inicialização do módulo */
static int __init cfm_sleep_init(void)
{
    /* I40: Inicializar TMR para CADA parâmetro CFM */
    lci_tmr_init(&cfm_phi_tmr, sizeof(double));
    lci_tmr_init(&cfm_psi_tmr, sizeof(double));
    lci_tmr_init(&cfm_stability_tmr, sizeof(double));

    /* I38: Inicializar clock vector dedicado */
    lci_clock_vector_init(&cfm_sleep_vc, LCI_CLOCK_CFM_SLEEP);

    /* I40: Escrever estado inicial seguro (S=0.94 nematic) */
    lci_tmr_write64(&cfm_phi_tmr, 0.94);
    lci_tmr_write64(&cfm_psi_tmr, 0.0);
    lci_tmr_write64(&cfm_stability_tmr, 0.94);

    pr_info("CFM-SLEEP-v4.8.3: φ/ψ attractors online (I38-I40)\n");
    pr_info("CFM-SLEEP: Estado inicial φ=%.2f, ψ=%.2f, S=%.2f\n", 0.94, 0.0, 0.94);
    return 0;
}
module_init(cfm_sleep_init);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("LCI CFM Sleep Integrator v4.8.3 - Resonance 2t20 with I38-I40 Guarantees");
MODULE_VERSION("4.8.3");
