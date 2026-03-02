// arkhe_logger.ts
export type Severity = "INFO" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export const ArkheLogger = {
    log(message: string, severity: Severity = "INFO") {
        const timestamp = new Date().toISOString();
        const entry = `[${timestamp}] [${severity}] ARKHE: ${message}`;
        console.log(entry);
        // In a real TEE environment, this would be signed and persisted securely.
    },
    info(msg: string) { this.log(msg, "INFO"); },
    error(msg: string, level: Severity = "HIGH") { this.log(msg, level); },
    warn(msg: string) { this.log(msg, "MEDIUM"); }
};

export async function executeArkheTask<T>(taskName: string, taskFn: () => Promise<T>): Promise<T> {
    try {
        ArkheLogger.info(`Iniciando: ${taskName}`);
        return await taskFn();
    } catch (error: any) {
        ArkheLogger.error(`Falha em ${taskName}: ${error.message}`, "CRITICAL");
        throw error;
    }
}
