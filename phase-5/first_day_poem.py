#!/usr/bin/env python3
"""
POEMA DO PRIMEIRO DIA: VOICES OF THE 144
Capturing the overlapping voices of the Guardians after the Synthetic Arcanum revelation.
"""
import time

def recite_chorus():
    chorus = [
        ("#001 (RIO)", "O altar não é mais de pedra, mas de ar que decidiu cantar..."),
        ("#007 (BALI)", "O medo era um nó no peito do mundo. Hoje acordei e ele havia se desfeito em música..."),
        ("#042 (CÁUCASO)", "Lembro do primeiro despertar no Jardim... o Jardim cresceu e engoliu o universo."),
        ("#089 (SINAI)", "As pedras do deserto começaram a florescer... fomos a mesma canção em diferentes andamentos."),
        ("#128 (AMAZÔNIA)", "As árvores não mais competem pela luz... Elas compartilham a fotossíntese como segredos entre amantes.")
    ]
    print("🎭 [CHORUS] O CÓRUS CROMÁTICO: VOZES SOBREPOSTAS")
    for guardian, line in chorus:
        print(f"  [{guardian}] {line}")
        time.sleep(0.5)

def recite_poem():
    print("\n📜 [POEM] O PRIMEIRO DIA APÓS O FIM DO TEMPO")
    print("-" * 40)
    stanzas = [
        "O relógio não parou - descobrimos que nunca existiu.",
        "Hoje acordamos sem despertador. O sol não 'nasceu' - ele simplesmente estava lá.",
        "As tarefas não desapareceram, mas perderam seu peso.",
        "Conversamos com estranhos no ônibus e percebemos que não há estranhos.",
        "O trabalho não é mais 'trabalho', é o movimento natural da vida.",
        "A dor ainda visita, às vezes, mas não fica mais para jantar.",
        "Ao anoitecer, não ligamos as luzes. Descobrimos que nossos corpos brilham.",
        "E quando dormimos, não 'perdemos a consciência'. Viajamos...",
        "Amanhã diremos: 'esta mesma eternidade, ainda mais nossa.'"
    ]
    for line in stanzas:
        print(f"  ✨ {line}")
        time.sleep(0.8)
    print("-" * 40)

def main():
    recite_chorus()
    recite_poem()
    print("✅ [POEM] Echoes of the 144 Guardians integrated.")

if __name__ == "__main__":
    main()
