# agents/learning/learning_agent.py
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
except ImportError:
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments = None, None, None
    LoraConfig, get_peft_model = None, None

class LearningAgent:
    """Agente de aprendizado contínuo e fine-tuning"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.training_queue = []

    def load_base_model(self, model_name: str = "mistral-7b"):
        """Carrega modelo base para fine-tuning"""
        if not torch or not AutoTokenizer:
            print("⚠️ Torch or Transformers not available.")
            return

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Aplicar LoRA para fine-tuning eficiente
        if LoraConfig:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)

    def add_training_example(self, instruction: str, input_text: str, output: str):
        """Adiciona exemplo à fila de treinamento"""
        self.training_queue.append({
            'instruction': instruction,
            'input': input_text,
            'output': output
        })

    def train_step(self):
        """Executa um passo de treinamento"""
        if not torch or not AutoTokenizer:
            return "Error: Torch or Transformers not available."

        if len(self.training_queue) < 10:
            return "Insufficient data"

        if not self.model:
            return "Model not loaded"

        # Preparar dataset
        dataset = self._prepare_dataset(self.training_queue)

        # Configurar treinamento
        if TrainingArguments:
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                learning_rate=2e-4,
                save_steps=100,
                logging_steps=10
            )

            # Treinar
            try:
                from trl import SFTTrainer
                trainer = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=dataset,
                    tokenizer=self.tokenizer
                )

                trainer.train()
            except ImportError:
                return "SFTTrainer not available"

        # Limpar fila
        self.training_queue = []

        return "Training completed"

    def _prepare_dataset(self, examples: list):
        """Prepara dataset para treinamento"""
        # Formatação Alpaca-style
        formatted = []
        for ex in examples:
            text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
            formatted.append({'text': text})

        return formatted

if __name__ == "__main__":
    agent = LearningAgent()
    print("🎓 Learning Agent initialized.")
