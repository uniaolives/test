# arkhe_web3/story_protocol.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
import json

@dataclass
class RoyaltyPolicy:
    """
    Defines how royalties are distributed for an IP Asset.
    """
    policy_id: str
    royalty_basis_points: int  # e.g., 500 for 5%
    distribution_model: str = "linear"  # "linear", "recursive", "topological"

@dataclass
class IPAsset:
    """
    Representação de um Ativo de Propriedade Intelectual (IPA) no Story Protocol.
    """
    asset_id: str = ""
    title: str = ""
    owner: str = ""
    metadata_hash: str = ""
    ip_type: str = "patent"  # patent, software, research, artwork

    # Story Protocol Specifics
    parent_id: Optional[str] = None
    royalty_policy: Optional[RoyaltyPolicy] = None
    is_commercial: bool = False

    # Arkhe(n) Coherence Metrics
    phi_score: float = 0.0
    coherence: float = 1.0
    registered_at: datetime = field(default_factory=datetime.now)

    def compute_asset_hash(self) -> str:
        """Calcula o hash único do ativo para o ledger."""
        content = {
            'title': self.title,
            'owner': self.owner,
            'metadata': self.metadata_hash,
            'ip_type': self.ip_type,
            'parent': self.parent_id,
            'timestamp': self.registered_at.isoformat()
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

@dataclass
class License:
    """
    Licença concedida para uso de um IP Asset.
    """
    license_id: str
    asset_id: str
    licensee: str
    terms_hash: str
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    active: bool = True

class StoryManifold:
    """
    Manifold para gestão de Propriedade Intelectual Programável (Story Protocol).
    Garante que C + F = 1 na gestão de direitos e royalties.
    """

    def __init__(self):
        self.assets: Dict[str, IPAsset] = {}
        self.licenses: Dict[str, License] = {}
        self.royalty_vault: Dict[str, int] = {} # asset_id -> satoshi balance (integer)
        self.omega_ledger: List[dict] = []
        self.PHI = 0.618033988749895

    def register_ip_asset(self, asset: IPAsset) -> str:
        """
        Registra um novo IP Asset no Story Protocol via Arkhe.
        """
        asset_hash = asset.compute_asset_hash()
        asset.asset_id = f"IPA-{asset_hash[:12]}"

        self.assets[asset.asset_id] = asset
        self.royalty_vault[asset.asset_id] = 0

        # Registro no Omega Ledger
        self.omega_ledger.append({
            'type': 'IPA_REGISTERED',
            'timestamp': datetime.now().isoformat(),
            'asset_id': asset.asset_id,
            'owner': asset.owner,
            'ip_type': asset.ip_type,
            'phi_score': asset.phi_score
        })

        return asset.asset_id

    def create_license(self, asset_id: str, licensee: str, terms: dict) -> str:
        """
        Gera uma licença para um ativo existente.
        """
        if asset_id not in self.assets:
            raise ValueError("Asset not found")

        terms_hash = hashlib.sha256(json.dumps(terms, sort_keys=True).encode()).hexdigest()
        lic_id = f"LIC-{terms_hash[:12]}"

        license = License(
            license_id=lic_id,
            asset_id=asset_id,
            licensee=licensee,
            terms_hash=terms_hash
        )

        self.licenses[lic_id] = license

        # Log de transação
        self.omega_ledger.append({
            'type': 'LICENSE_GRANTED',
            'timestamp': datetime.now().isoformat(),
            'license_id': lic_id,
            'asset_id': asset_id,
            'licensee': licensee
        })

        return lic_id

    def collect_royalty(self, asset_id: str, amount: int):
        """
        Coleta royalties e distribui de acordo com a política (Simulado).
        """
        if asset_id not in self.assets:
            return

        asset = self.assets[asset_id]
        if not asset.royalty_policy:
            self.royalty_vault[asset_id] += amount
            return

        # Distribuição baseada na política (usando basis points para evitar floats)
        fee = (amount * asset.royalty_policy.royalty_basis_points) // 10000
        net = amount - fee

        self.royalty_vault[asset_id] += net

        # Se houver parent_id, repassa parte do fee recursivamente (Derivativos IP)
        if asset.parent_id and asset.parent_id in self.assets:
            self.collect_royalty(asset.parent_id, fee)

    def get_asset_coherence(self, asset_id: str) -> float:
        """
        Calcula a coerência do ativo baseada no uso e royalties.
        """
        if asset_id not in self.assets:
            return 0.0

        asset = self.assets[asset_id]
        n_licenses = len([l for l in self.licenses.values() if l.asset_id == asset_id])
        revenue = float(self.royalty_vault.get(asset_id, 0))

        # Coerência aumenta com uso legítimo (licenças) e retorno (revenue)
        usage_factor = np.tanh(n_licenses / 10.0)
        revenue_factor = np.tanh(revenue / 100.0)

        coherence = (asset.phi_score * self.PHI) + ((usage_factor + revenue_factor) / 2.0 * (1.0 - self.PHI))
        return float(min(1.0, coherence))

if __name__ == "__main__":
    manifold = StoryManifold()

    # Exemplo: Registrar a Patente de Anyons
    anyon_patent = IPAsset(
        title="Anyonic Phase Accumulation for Space Communication",
        owner="Rafael Oliveira",
        metadata_hash="hash_whitepaper_2026",
        ip_type="patent",
        phi_score=0.965,
        royalty_policy=RoyaltyPolicy("POL-001", 500) # 500 basis points = 5%
    )

    ipa_id = manifold.register_ip_asset(anyon_patent)
    print(f"Registered IP Asset: {ipa_id}")

    # Criar licença para uma operadora de satélites
    lic_id = manifold.create_license(ipa_id, "Starlink-Arkhe-Branch", {"use": "orbital_comms", "region": "LEO"})
    print(f"License Created: {lic_id}")

    # Coletar Royalties
    manifold.collect_royalty(ipa_id, 5000) # 5000 Satoshis
    print(f"Vault Balance for {ipa_id}: {manifold.royalty_vault[ipa_id]} Satoshi")
    print(f"Asset Coherence: {manifold.get_asset_coherence(ipa_id):.3f}")
    print("∞")
