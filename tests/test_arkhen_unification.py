# Integrated Unification Test for ArkheOS Convergence Architecture
import asyncio
from arkhe.kernel import DocumentIngestor, AnchorResolver
from arkhe.extraction import GeminiExtractor, Provenance, Currency
from arkhe.registry import GlobalEntityRegistry, EntityCandidate, EntityType
from arkhe.consensus import GeodesicConsensus, ValidatedFact, ConsensusStatus

import pytest
from arkhe.kernel import DocumentIngestor, AnchorResolver
from arkhe.extraction import GeminiExtractor, Provenance, Currency, ExtractionReport
from arkhe.registry import GlobalEntityRegistry, EntityCandidate, EntityType
from arkhe.consensus import GeodesicConsensus, ValidatedFact, ConsensusStatus

@pytest.mark.asyncio
async def test_arkhen_convergence():
    print("🚀 Starting ArkheOS Integrated Convergence Test...")

    # 1. Physical Ingestion (Kernel)
    ingestor = DocumentIngestor(provider="local")
    layout_elements = ingestor.process("annual_report_2025.pdf")
    print(f"   [Kernel] Ingested {len(layout_elements)} physical layout elements.")

    # 2. Parallel Extraction (Extraction)
    # Model A
    ext_a = GeminiExtractor(api_key="sk-alpha")
    report_a = ext_a.extract("...net profit of $1.2M...", "hash-123", 1, "annual_report_2025.pdf")
    report_a = await ext_a.extract("...net profit of $1.2M...", ExtractionReport)

    # Model B (Simulated Divergence)
    fact_b = report_a.facts[0].model_copy()
    fact_b.value = 1100000.0 # Divergent value

    # 3. State Reconciliation (Registry)
    registry = GlobalEntityRegistry()

    cand_a = EntityCandidate(
        name="net_profit",
        entity_type=EntityType.FINANCIAL,
        value=report_a.facts[0].value,
        unit="USD",
        confidence=0.98,
        provenance=report_a.facts[0].provenance,
        chunk_id="c1"
    )

    cand_b = EntityCandidate(
        name="net_profit",
        entity_type=EntityType.FINANCIAL,
        value=fact_b.value,
        unit="USD",
        confidence=0.95,
        provenance=report_a.facts[0].provenance,
        chunk_id="c2"
    )

    ent, conflicted = registry.ingest_candidate(cand_a)
    ent, conflicted = registry.ingest_candidate(cand_b)

    print(f"   [Registry] Reconciled entity '{ent.name}'. Conflicted: {conflicted}")
    assert conflicted == True

    # 4. Multi-Model Consensus (Consensus)
    consensus = GeodesicConsensus()
    validated = consensus.reconcile(report_a.facts[0], fact_b)
    print(f"   [Consensus] Status: {validated.consensus_status}")
    assert validated.consensus_status == ConsensusStatus.DIVERGED

    # 5. Manual Resolution (Identity Stone)
    registry.resolve_manually(ent.id, 1200000.0, "Audited table takes precedence")
    print(f"   [Identity] Resolved manually. State: {ent.state}")
    assert ent.state == "confirmed"

    print("✅ Convergence Protocol Verified (Φ = 1.0000).")

if __name__ == "__main__":
    asyncio.run(test_arkhen_convergence())
