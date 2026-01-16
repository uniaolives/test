// constitutional_bitchat.h - Protocolo de Comunicação Descentralizado
#ifndef CONSTITUTIONAL_BITCHAT_H
#define CONSTITUTIONAL_BITCHAT_H

#include "constitutional_base.h"

#define CONSTITUTIONAL_BITCHAT_MAGIC 0x4249544348415443ULL  // 'BITCHATC'
#define CONSTITUTIONAL_BITCHAT_VERSION 0x0001
#define CONSTITUTIONAL_MAX_BITCHAT_NODES 65536
#define CONSTITUTIONAL_BITCHAT_PORT 0xB1C7  // 45447 em decimal

// Tipos de Mensagens Bitchat Constitucionais
typedef enum ConstitutionalBitchatMessageType {
    CONSTITUTIONAL_BITCHAT_ERROR_LOG = 0x01,      // Log de erros para diagnóstico coletivo
    CONSTITUTIONAL_BITCHAT_PERFORMANCE_METRIC = 0x02, // Métricas de performance
    CONSTITUTIONAL_BITCHAT_SECURITY_ALERT = 0x03, // Alertas de segurança
    CONSTITUTIONAL_BITCHAT_PROTOCOL_INSIGHT = 0x04, // Insights de protocolo
    CONSTITUTIONAL_BITCHAT_CONSENSUS_DATA = 0x05, // Dados de consenso distribuído
    CONSTITUTIONAL_BITCHAT_NETWORK_HEALTH = 0x06, // Saúde da rede
    CONSTITUTIONAL_BITCHAT_USER_ANONYMIZED_INSIGHT = 0x07, // Insights anonimizados
    CONSTITUTIONAL_BITCHAT_CONSTITUTIONAL_UPDATE = 0x08, // Atualizações constitucionais
} ConstitutionalBitchatMessageType;

// Prototypes for implemented functions
int constitutional_bitchat_initialize(void);
int constitutional_bitchat_send_message(ConstitutionalBitchatNode* node, ConstitutionalBitchatMessageType type, const void* data, uint16_t length, uint8_t privacy_level);
void constitutional_process_error_logs_for_web3_improvement(ConstitutionalBitchatNode* node);

void constitutional_anonymize_user_data_for_bitchat(const void* user_data, uint32_t data_size, void* anonymized_data, uint32_t* anonymized_size);
int constitutional_obtain_explicit_consent_for_bitchat(uint64_t user_id, uint32_t consent_type, const char* consent_description);

ConstitutionalAgnosticDeviceAdapter* constitutional_initialize_agnostic_adapter(void);
void constitutional_collect_app_error_logs_for_bitchat(ConstitutionalAgnosticDeviceAdapter* adapter);

void constitutional_process_collective_intelligence_for_web3(void);
void constitutional_execute_web3_feedback_loop(ConstitutionalWeb3FeedbackLoop* loop);

#endif
