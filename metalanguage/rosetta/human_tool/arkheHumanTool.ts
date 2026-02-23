// arkheHumanTool.ts
// Reference implementation for Arkhe(n) Human-Tool Interface in TypeScript

interface Human {
    processingCapacity: number;  // bits/min
    attentionSpan: number;       // minutes
    currentLoad: number;
    goals: string[];
}

interface Tool {
    outputVolume: number;        // tokens/min
    outputEntropy: number;       // bits/token
    hasDiscernment: boolean;
    hasIntentionality: boolean;
    hasPerception: boolean;
}

interface LogEntry {
    time: number;
    event: string;
    load?: number;
    intent?: string;
    approved?: boolean;
    output?: string;
}

class InteractionGuard {
    private human: Human;
    private tool: Tool;
    private log: LogEntry[] = [];
    private threshold: number = 0.7;

    constructor(human: Human, tool: Tool) {
        this.human = human;
        this.tool = tool;
    }

    proposeInteraction(intent: string): string | null {
        const load = (this.tool.outputVolume * this.tool.outputEntropy) / this.human.processingCapacity;

        if (load > this.threshold) {
            this.log.push({
                time: Date.now(),
                event: 'BLOCKED',
                load: load
            });
            return null;
        }

        if (this.human.currentLoad > 0.8) {
            this.log.push({
                time: Date.now(),
                event: 'BLOCKED',
                load: this.human.currentLoad
            });
            return null;
        }

        // Simular geração (em produção, chamaria API)
        const output = `Generated content for: ${intent}`;

        const impact = load * 0.3;
        this.human.currentLoad = Math.min(1.0, this.human.currentLoad + impact);

        this.log.push({
            time: Date.now(),
            event: 'GENERATED',
            load: load,
            intent: intent
        });

        return output;
    }

    review(output: string, approved: boolean): void {
        this.log.push({
            time: Date.now(),
            event: 'REVIEWED',
            approved: approved,
            output: output.substring(0, 100)
        });
        if (approved) {
            this.human.currentLoad = Math.max(0, this.human.currentLoad - 0.1);
        }
    }

    cognitiveLoadIndex(windowMinutes: number = 60): number {
        const cutoff = Date.now() - windowMinutes * 60 * 1000;
        const recent = this.log.filter(e => e.time > cutoff);
        const overloads = recent.filter(e => (e.load || 0) > this.threshold);
        return overloads.length / Math.max(1, recent.length);
    }

    authorshipLossRate(windowMinutes: number = 60): number {
        const cutoff = Date.now() - windowMinutes * 60 * 1000;
        const recent = this.log.filter(e => e.time > cutoff);
        const reviews = recent.filter(e => e.event === 'REVIEWED').length;
        const total = recent.filter(e => e.event === 'GENERATED' || e.event === 'REVIEWED').length;
        return reviews / Math.max(1, total);
    }
}

export { Human, Tool, InteractionGuard };
