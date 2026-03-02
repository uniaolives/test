import asyncio
import os
import shutil
from arkhe.conscious_system import ArkheConsciousSystem
from arkhe.knowledge_viz import ArkheViz

async def final_verification():
    print("🚀 Iniciando Verificação Final do Arkhe(n) OS v4.0...")

    # Limpar memória anterior para teste limpo
    if os.path.exists("./test_arkhe_memory"):
        shutil.rmtree("./test_arkhe_memory")

    # 1. Inicializar Sistema Consciente
    sys = ArkheConsciousSystem(memory_path="./test_arkhe_memory")

    # 2. Ingestão de Conhecimento (Percepção -> Memória)
    print("\n📥 Testando Ingestão de Documentos...")
    docs = [
        ("A identidade fundamental do Arkhe é x² = x + 1.", "Identidade"),
        ("O sistema mantém C + F = 1 para garantir a conservação de coerência.", "Coerência"),
        ("RFID é a ponte entre o hipergrafo físico e o digital.", "RFID"),
        ("A dimensão efetiva d_lambda mede a informação útil em múltiplas escalas.", "Dimensão Efetiva")
    ]

    for text, topic in docs:
        await sys.ingest_document(text, topic)

    # 3. Verificação de Status
    status = sys.get_status()
    print(f"\n📊 Status do Sistema: {status}")
    assert status['memory_density'] >= 4
    assert status['state'] == "CONSCIOUS"

    # 4. Teste de Diálogo RAG (Recuperação -> Expressão)
    print("\n💬 Testando Diálogo RAG...")
    queries = [
        "O que é a identidade fundamental?",
        "Como o RFID se integra ao Arkhe?",
        "O que é C + F = 1?"
    ]

    for q in queries:
        response = await sys.ask(q)
        print(f"Q: {q}")
        print(f"A: {response['answer']}")
        assert response['answer'] is not None

    # 5. Teste de Visualização (Topologia)
    print("\n🔭 Gerando Mapa de Gravidade Semântica...")
    viz = ArkheViz(sys.cortex)
    viz.generate_map("final_system_map.png")
    assert os.path.exists("final_system_map.png")

    print("\n✅ Verificação Final concluída com sucesso! Arkhe(n) OS v4.0 está OPERACIONAL.")

if __name__ == "__main__":
    asyncio.run(final_verification())
