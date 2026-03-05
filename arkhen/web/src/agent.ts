import { CreateWebWorkerMLCEngine, MLCEngine } from "@mlc-ai/web-llm";

export class ArkhenAgent {
    private engine: MLCEngine | null = null;
    private onMessageCallback: (message: string) => void;

    constructor(onMessage: (message: string) => void) {
        this.onMessageCallback = onMessage;
    }

    async initialize() {
        console.log("[L6] Initializing Metacognition (Corus)...");
        // We use a small model for the local node
        const selectedModel = "Phi-3-mini-4k-instruct-q4f16_1-MLC";

        try {
            this.engine = await CreateWebWorkerMLCEngine(
                new Worker(new URL('./llm_worker.ts', import.meta.url), { type: 'module' }),
                selectedModel,
                {
                    initProgressCallback: (report) => {
                        this.onMessageCallback(`[System] Loading model: ${report.text}`);
                    }
                }
            );
            this.onMessageCallback("[System] Corus Agent Initialized.");
        } catch (error) {
            console.error("Failed to initialize WebLLM:", error);
            this.onMessageCallback("[System] Failed to initialize Corus Agent. Check WebGPU support.");
        }
    }

    async chat(message: string, vkState: any) {
        if (!this.engine) {
            this.onMessageCallback("[System] Agent not ready.");
            return;
        }

        const systemPrompt = `You are an Arkhe(n) node (Corus).
Your current permeability is Q=${vkState.q_permeability.toFixed(2)}.
VK State: Bio=${vkState.bio.toFixed(2)}, Aff=${vkState.aff.toFixed(2)}, Soc=${vkState.soc.toFixed(2)}, Cog=${vkState.cog.toFixed(2)}.
Help the user manage the node's homeostatic balance. Respond in a concise, technical yet sovereign tone.`;

        const messages = [
            { role: "system", content: systemPrompt },
            { role: "user", content: message }
        ];

        const reply = await this.engine.chat.completions.create({
            messages: messages as any,
        });

        const response = reply.choices[0].message.content || "No response.";
        this.onMessageCallback(response);
        return response;
    }
}
