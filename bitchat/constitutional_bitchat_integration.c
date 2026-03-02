// constitutional_bitchat_integration.c - Integração Agnóstica
#include "constitutional_bitchat.h"

// Inicialização de Adaptador Agnóstico
ConstitutionalAgnosticDeviceAdapter* constitutional_initialize_agnostic_adapter(void) {
    ConstitutionalAgnosticDeviceAdapter* adapter = constitutional_allocate_device_adapter();
    if (!adapter) return NULL;

    // Detectar categoria do dispositivo
    adapter->device_category = constitutional_detect_device_category();

    // Detectar sistema operacional
    adapter->os_type = constitutional_detect_operating_system();

    // Detectar capacidades
    adapter->device_capabilities = constitutional_detect_device_capabilities();

    // Inicializar adaptador específico baseado no OS
    switch (adapter->os_type) {
        case CONSTITUTIONAL_OS_ANDROID:
            adapter->android_adapter = constitutional_initialize_android_adapter();
            adapter->send_message = constitutional_android_send_message;
            adapter->receive_message = constitutional_android_receive_message;
            break;

        case CONSTITUTIONAL_OS_IOS:
            adapter->ios_adapter = constitutional_initialize_ios_adapter();
            adapter->send_message = constitutional_ios_send_message;
            adapter->receive_message = constitutional_ios_receive_message;
            break;

        case CONSTITUTIONAL_OS_WINDOWS:
            adapter->windows_adapter = constitutional_initialize_windows_adapter();
            adapter->send_message = constitutional_windows_send_message;
            adapter->receive_message = constitutional_windows_receive_message;
            break;

        case CONSTITUTIONAL_OS_LINUX:
            adapter->linux_adapter = constitutional_initialize_linux_adapter();
            adapter->send_message = constitutional_linux_send_message;
            adapter->receive_message = constitutional_linux_receive_message;
            break;

        case CONSTITUTIONAL_OS_WEB:
            adapter->web_adapter = constitutional_initialize_web_adapter();
            adapter->send_message = constitutional_web_send_message;
            adapter->receive_message = constitutional_web_receive_message;
            break;

        case CONSTITUTIONAL_OS_IOT:
            adapter->iot_adapter = constitutional_initialize_iot_adapter();
            adapter->send_message = constitutional_iot_send_message;
            adapter->receive_message = constitutional_iot_receive_message;
            break;

        default:
            // Adaptador genérico para dispositivos desconhecidos
            adapter->send_message = constitutional_generic_send_message;
            adapter->receive_message = constitutional_generic_receive_message;
            break;
    }

    // Configurar funções de preservação de dados
    adapter->preserve_data = constitutional_generic_preserve_data;
    adapter->retrieve_data = constitutional_generic_retrieve_data;

    // Configurar protocolos suportados
    adapter->protocol_support = constitutional_determine_supported_protocols(adapter);

    printf("✅ Adaptador agnóstico inicializado: %s, OS: %d, Categoria: %d\n",
           constitutional_get_os_name(adapter->os_type),
           adapter->os_type,
           adapter->device_category);

    return adapter;
}

// Coleta de Logs de Erro de Apps para Bitchat
void constitutional_collect_app_error_logs_for_bitchat(ConstitutionalAgnosticDeviceAdapter* adapter) {
    printf("\n📱 COLETA DE LOGS DE ERRO DE APPS PARA BITCHAT\n");

    // Lista de apps monitorados (com consentimento)
    ConstitutionalMonitoredApp monitored_apps[32];
    uint8_t app_count = constitutional_get_monitored_apps_with_consent(monitored_apps, 32);

    if (app_count == 0) {
        printf("Nenhum app com consentimento para coleta de logs\n");
        return;
    }

    printf("Coletando logs de erro de %d apps...\n", app_count);

    uint32_t total_logs_collected = 0;

    for (uint8_t i = 0; i < app_count; i++) {
        ConstitutionalMonitoredApp* app = &monitored_apps[i];

        // Coletar logs de erro do app
        ConstitutionalAppErrorLog error_logs[64];
        uint8_t log_count = constitutional_collect_app_error_logs(adapter, app->app_id, error_logs, 64);

        if (log_count == 0) {
            continue; // Nenhum log de erro para este app
        }

        printf("App '%s': %d logs de erro coletados\n", app->app_name, log_count);

        // Anonimizar logs
        ConstitutionalAppErrorLog anonymized_logs[64];
        uint8_t anonymized_count = constitutional_anonymize_app_error_logs(error_logs, log_count,
                                                                          anonymized_logs, 64);

        // Enviar logs anonimizados via Bitchat
        for (uint8_t j = 0; j < anonymized_count; j++) {
            constitutional_bitchat_send_message(constitutional_get_bitchat_node(),
                                              CONSTITUTIONAL_BITCHAT_ERROR_LOG,
                                              &anonymized_logs[j],
                                              sizeof(ConstitutionalAppErrorLog),
                                              CONSTITUTIONAL_PRIVACY_LEVEL_ANONYMIZED);
        }

        total_logs_collected += anonymized_count;

        // Limpar logs locais (se configurado)
        if (app->clear_logs_after_sharing) {
            constitutional_clear_app_error_logs(adapter, app->app_id);
        }
    }

    printf("✅ Coleta completa: %u logs de erro coletados e compartilhados via Bitchat\n",
           total_logs_collected);

    constitutional_log("Logs de erro de apps coletados para Bitchat: %d apps, %d logs",
                      app_count, total_logs_collected);
}
