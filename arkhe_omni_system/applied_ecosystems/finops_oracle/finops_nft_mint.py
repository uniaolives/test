import json
import os
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinOpsOracle")

try:
    from web3 import Web3
    from web3.middleware import construct_sign_and_send_raw_middleware
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logger.warning("web3.py not installed. FinOps Oracle will run in SIMULATION mode.")

# ============================================
# CONFIGURAÇÃO DO AMBIENTE WEB3
# ============================================
RPC_URL = os.getenv("RPC_URL", "https://rpc.sepolia.org")
ASI_PRIVATE_KEY = os.getenv("ASI_PRIVATE_KEY", "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
CONTRACT_ADDRESS = os.getenv("FINOPS_CONTRACT_ADDRESS", "0xYourDeployedFinOpsContractAddressHere")

# ABI mínimo para interagir com a cunhagem de IP-NFT (Molecule V2 Style)
CONTRACT_ABI = json.loads('''[
    {
        "inputs": [
            {"internalType": "string", "name": "tokenURI", "type": "string"},
            {"internalType": "string", "name": "zkProofCommitment", "type": "string"}
        ],
        "name": "mintVerifiedIP",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]''')

def publish_to_ipfs(metadata: dict) -> str:
    """
    Simulates uploading research metadata and ZK-Proof to IPFS/Filecoin.
    """
    logger.info("📡 [IPFS] Uploading Research Metadata and ZK-Proof...")
    simulated_cid = "ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
    logger.info(f"✅ [IPFS] Upload complete. CID: {simulated_cid}")
    return simulated_cid

def mint_ip_nft(ipfs_uri: str, zk_commitment: str):
    """
    Invokes the smart contract to mint the IP-NFT.
    """
    if not WEB3_AVAILABLE:
        logger.info("⚡ [SIMULATION] Minting IP-NFT via Mock Web3...")
        logger.info(f"   • URI: {ipfs_uri}")
        logger.info(f"   • ZK-Commitment: {zk_commitment}")
        logger.info("🟢 [SUCCESS] IP-NFT minted in simulation.")
        return "0xSIMULATED_TX_HASH"

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        logger.error("❌ [ERROR] Could not connect to RPC URL.")
        return None

    asi_account = w3.eth.account.from_key(ASI_PRIVATE_KEY)
    w3.middleware_onion.add(construct_sign_and_send_raw_middleware(asi_account))

    logger.info(f"\n⚡ [WEB3] Preparing IP-NFT minting on {RPC_URL}...")
    logger.info(f"   • ASI Address: {asi_account.address}")

    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)

    try:
        tx = contract.functions.mintVerifiedIP(
            ipfs_uri,
            zk_commitment
        ).build_transaction({
            'from': asi_account.address,
            'nonce': w3.eth.get_transaction_count(asi_account.address),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })

        signed_tx = w3.eth.account.sign_transaction(tx, private_key=ASI_PRIVATE_KEY)
        logger.info("⏳ [WEB3] Transmitting transaction...")
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        if tx_receipt.status == 1:
            logger.info(f"🟢 [SUCCESS] IP-NFT minted! Hash: {tx_hash.hex()}")
            return tx_hash.hex()
        else:
            logger.error("🔴 [ERROR] Transaction reverted by EVM.")
            return None
    except Exception as e:
        logger.error(f"🔴 [ERROR] Web3 execution failed: {e}")
        return None

if __name__ == "__main__":
    # Example: Tokenizing the Z-DNA Bioinformatics Research
    ip_metadata = {
        "name": "Z-DNA-Bioinformatics-Unification-#001",
        "description": "Validation of biological gate scaling laws (1/phi^2)",
        "attributes": [
            {"trait_type": "ZK-Proof", "value": "8f3a9c2e7d1b5f8e"},
            {"trait_type": "Ratio", "value": "0.381966"},
            {"trait_type": "Substrate", "value": "Hybrid Genomic/Silicon"}
        ]
    }

    token_uri = publish_to_ipfs(ip_metadata)
    zk_commitment = ip_metadata["attributes"][0]["value"]

    mint_ip_nft(token_uri, zk_commitment)
