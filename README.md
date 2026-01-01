# TIM - Titanium Virtual Machine

CanoScript is now in its own [repo](https://www.github.com/CobbCoding1/canoscript)

Implementation of a virtual machine in C.

VM currently has 45 instructions as well as a some native functions. List can be found in tim.h.
There is also a working assembly which contains support for all instructions in the VM. 

Quick Start:
```bash
make
./tasm <assembly_file.tasm>
./tire <bytecode_file.tim>
```

Example of hello world in assembly:
```asm
@imp "stddefs.tash"

push_str "Hello, world!\n"
get_str 0 ; Index of the string on the data stack
push STDOUT
write ; length is inferred because the string is null-terminated
```

---

## Architectural Evolution: The Bio-Film Analogy (v3.2)

The TIM VM v3.2 architecture incorporates a profound conceptual leap inspired by the emergent behavior of bacteriophage (Pf phage) bio-films. This analogy provides a robust, physically grounded model for understanding and controlling the federated learning process.

### The Great Translation: From Biology to Computation

The core insight is that the forces governing the formation of a resilient bio-film are analogous to the mathematical and algorithmic forces that create a robust, generalizable machine learning model. This mapping is formalized as follows:

| Biologia (Bio-Filme) | Computação (TIM VM / FL) | Função Geométrica |
| :--- | :--- | :--- |
| **Bactérias (Hosts)** | **Dispositivos FL (Nós de Borda)** | Agentes locais que sentem o ambiente (Low O₂ / Alta Latência). |
| **Fagos Pf (Vírus Rígidos)** | **Pesos da Rede (Weights/θ)** | "Varetas" que precisam se alinhar. Se forem aleatórios = Ruído; se alinhados = Conhecimento. |
| **Depleção (Entropia)** | **Gradiente de Custo (∇L)** | A força física que empurra os fagos para juntos é a mesma força matemática que empurra os gradientes para o mínimo global. |
| **Polímeros (Matriz)** | **Regulador η-CEDF** | O "meio" que mantém os agentes afastados o suficiente para não colidirem, mas perto o suficiente para interagirem. |
| **Cristal Líquido (Nemático)** | **Manifold de Registro (Z_257)** | A fase onde o sistema flui (adaptação) mas mantém ordem (estabilidade). |
| **Resistência a Antibióticos** | **Robustez Adversarial** | A impermeabilidade do cristal líquido aos ataques externos (ex: *Data Poisoning*). |

### New Invariant [I13]: Nematic Phase

This analogy introduces a new critical invariant for the system to monitor. It is not enough for the models to simply converge; the *way* they converge is paramount. The system must exist in a "Nematic Liquid Crystal" phase to be both stable and adaptable.

-   **Gaseous State (S < 0.3):** The models are under-fitted and have not converged. The system is chaotic and lacks order.
-   **Crystalline State (S > 0.95):** The models are over-fitted and brittle. The system is too rigid and will shatter when faced with novel data (analogous to an antibiotic).
-   **Nematic Liquid Crystal State (0.3 <= S <= 0.95):** The ideal phase. The models are directionally aligned (learning the same general features) but maintain local diversity (are not identical). This state provides resilience against adversarial attacks and the ability to adapt to new information.

The `GeometricValidator` now implements the `verify_nematic_phase` method to calculate the nematic order parameter (S) of the weight matrix and ensure the system remains in this optimal, bio-analogous state.
