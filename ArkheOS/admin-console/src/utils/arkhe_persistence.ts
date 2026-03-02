// arkhe_persistence.ts
import { ArkheLogger } from './arkhe_logger';

export const ArkhePersistence = {
    save(nodes: any) {
        try {
            const data = JSON.stringify(nodes);
            localStorage.setItem('ARKHE_NODES_V4', data);
            ArkheLogger.info("Ledger persistido com sucesso.");
        } catch (e: any) {
            ArkheLogger.error("Falha na persistência: " + e.message, "CRITICAL");
        }
    },
    load() {
        const saved = localStorage.getItem('ARKHE_NODES_V4');
        if (saved) {
            ArkheLogger.info("Nós recuperados do Enclave de Memória.");
            return JSON.parse(saved);
        }
        return null;
    }
};
