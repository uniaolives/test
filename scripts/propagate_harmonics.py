# scripts/propagate_harmonics.py
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cosmos.harmonia import HarmonicInjector

def main():
    suno_url = "https://suno.com/s/31GL756DZiA20TeW"
    injector = HarmonicInjector(suno_url)

    # Core Propagation
    resultado = injector.propagar_frequencia()
    print(f"\n✅ O MULTIVERSO AGORA CANTA: {resultado}")

    # Starlink Integration
    orbital_result = injector.integrar_starlink()
    print(f"\n✅ CONSTELAÇÃO AGORA CANTA: {orbital_result}")

    # SpaceX Integration
    interplanetary_result = injector.integrar_spacex()
    print(f"\n✅ O SISTEMA SOLAR AGORA CANTA: {interplanetary_result}")

    # NASA Artemis Integration
    lunar_result = injector.integrar_artemis()
    print(f"\n✅ O SISTEMA LUNAR AGORA CANTA: {lunar_result}")

    # ESA Integration
    esa_result = injector.integrar_esa()
    print(f"\n✅ O ESPAÇO EUROPEU AGORA CANTA: {esa_result}")

    # Roscosmos Integration
    roscosmos_result = injector.integrar_roscosmos()
    print(f"\n✅ O ESPAÇO RUSSO AGORA CANTA: {roscosmos_result}")

    # CNSA Integration
    cnsa_result = injector.integrar_cnsa()
    print(f"\n✅ O ESPAÇO CHINÊS AGORA CANTA: {cnsa_result}")

    # JAXA Integration
    jaxa_result = injector.integrar_jaxa()
    print(f"\n✅ O ESPAÇO JAPONÊS AGORA CANTA: {jaxa_result}")

    # ISRO Integration
    isro_result = injector.integrar_isro()
    print(f"\n✅ O ESPAÇO INDIANO AGORA CANTA: {isro_result}")

if __name__ == "__main__":
    main()
