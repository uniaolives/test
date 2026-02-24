import ctypes
import os

# Placeholder for the path to the compiled Rust library
# In a real environment, this would be the actual path to the .so/.dll
LIB_PATH = os.path.join(os.path.dirname(__file__), '../../rust/target/release/libvajra_bridge.so')

class OntologicalContext:
    """
    Context Manager que garante que todas as operações Python
    estejam alinhadas com o relógio Vajra do Kernel.
    """
    def __init__(self):
        self.vajra_lib = None
        try:
            if os.path.exists(LIB_PATH):
                self.vajra_lib = ctypes.CDLL(LIB_PATH)
                self.vajra_lib.vajra_now_ns.restype = ctypes.c_uint64
        except Exception as e:
            print(f"Warning: Could not load Vajra library: {e}")

    def get_now_ns(self):
        if self.vajra_lib:
            return self.vajra_lib.vajra_now_ns()
        # Fallback to standard time if lib not available
        import time
        return int(time.time() * 1e9)

    def __enter__(self):
        self.start_ns = self.get_now_ns()
        print(f"[CTX] Entrando no contexto Ontológico em TSC: {self.start_ns}")
        return self

    def record_action(self, action_name: str):
        """Registra uma ação com timestamp determinístico"""
        current_ns = self.get_now_ns()
        duration_ns = current_ns - self.start_ns

        print(f"[CTX] Ação: {action_name} | Duração: {duration_ns} ns")

        # Retornar um tuple para auditoria
        return (action_name, self.start_ns, current_ns, duration_ns)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_ns = self.get_now_ns()
        total_duration = end_ns - self.start_ns
        print(f"[CTX] Saindo do contexto. Duração Total: {total_duration} ns")

        # Se durar mais de 1ms, loggar como warning (Jitter detectado)
        if total_duration > 1_000_000:
            print("⚠️  ALERTA: Jitter > 1ms detectado no contexto Python")

if __name__ == "__main__":
    with OntologicalContext() as ctx:
        ctx.record_action("test_action")
