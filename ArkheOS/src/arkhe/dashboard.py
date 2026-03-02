# arkhe/dashboard.py
from .telemetry import TelemetryCollector

class MetricsDashboard:
    """
    Visualização das métricas C/F em tempo real.
    """

    def render(self, telemetry: TelemetryCollector):
        stats = telemetry.get_stats()

        print("\n" + "╔" + "═" * 60 + "╗")
        print("║" + "      ARKHE(n) TELEMETRY DASHBOARD v2.0".center(60) + "║")
        print("╠" + "═" * 60 + "╣")

        for provider, data in stats.items():
            # Usar disponibilidade ou taxa de sucesso como proxy de C
            C = data.get("availability", 0.0)
            F = 1.0 - C

            bar_len = 20
            filled_c = int(C * bar_len)
            bar_c = "█" * filled_c + "░" * (bar_len - filled_c)

            filled_f = int(F * bar_len)
            bar_f = "█" * filled_f + "░" * (bar_len - filled_f)

            print(f"║ {provider.upper():12} │ C: {bar_c} {C:.2f} │ F: {bar_f} {F:.2f} ║")
            print(f"║             │ Latency: {data.get('avg_latency_ms', 0):.0f}ms │ Calls: {data.get('total_calls', 0):<5} ║")
            print("╠" + "─" * 60 + "╣")

        print("║ Conservation Law: C + F = 1.0 ✅".center(60) + "║")
        print("╚" + "═" * 60 + "╝\n")

if __name__ == "__main__":
    from .telemetry import LLMMetrics, Provider
    import asyncio

    async def demo():
        collector = TelemetryCollector()
        await collector.record(LLMMetrics(Provider.GEMINI, "gen", 120.0, True, 10, 50))
        await collector.record(LLMMetrics(Provider.OLLAMA, "gen", 850.0, True, 10, 40))
        await collector.record(LLMMetrics(Provider.GEMINI, "gen", 150.0, False, 10, 0))

        dashboard = MetricsDashboard()
        dashboard.render(collector)

    asyncio.run(demo())
