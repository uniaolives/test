# Onionizing the Tzinor: Tor Transport Privacy in ArkheNet

## Overview
Application-layer privacy (ZK-proofs, shielded pools) is only as strong as the transport layer underneath it. ArkheNet implements a full-stack Tor integration to mask the physical locations and IP addresses of Bio-Nodes during inter-agent synchronization and transaction broadcasting.

## Core Pillars

### 1. Write-Side Privacy (Transaction Broadcasting)
When a Bio-Node emits an Orb to the Akasha L1 or any public blockchain, it routes the broadcast through a Tor circuit. This prevents RPC providers or mempool observers from linking a transaction to a specific IP address.

### 2. Read-Side Privacy (RPC & Data Retrieval)
Queries for blockchain state or remote telemetry are proxied through Tor, ensuring that behavioral data cannot be harvested by centralized RPC endpoints.

### 3. Censorship Resistance (Snowflake)
In restrictive network environments where Tor is blocked, ArkheNet utilizes pluggable transports like **Snowflake**. This disguises Tor traffic as innocuous WebRTC video/voice calls, bypassing deep packet inspection.

## Implementation Details
- **Embedded Arti**: ArkheOS embeds a native Tor client using the `arti-client` crate, eliminating the need for an external Tor daemon.
- **Onion Identities**: Bio-Nodes are assigned `.onion` addresses as their network-layer identities. These are decoupled from physical IPs and are authenticated via Ed25519 signatures.
- **Double-Encrypted Tunnels**: SSH channels (Tzinor) are multiplexed over Tor onion services, providing both application-layer authentication and network-layer anonymity.

## Architectural Equivalence
The onion routing protocol mirrors the **Kaluza-Klein compactification** used in Arkhe's topological framework. Each layer of encryption corresponds to a mode (ℤ) that is peeled off at each relay hop, until the payload eventually collapses into ℝ⁴ at the destination or exit point.

## Validator-Relay Synergy
Akasha validators are encouraged to run Tor non-exit relays. This creates a positive-sum feedback loop:
- **Validators** strengthen the Tor network by providing high-bandwidth middle nodes.
- **ArkheNet** gains a more resilient and anonymous transport layer for Orb propagation.
