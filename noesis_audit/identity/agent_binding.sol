// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title AgentIdentityRegistry
 * @dev Vincula cada agente autônomo a um humano verificável (Modelo Mandate).
 */
contract AgentIdentityRegistry {
    struct AgentBinding {
        address humanOwner;        // carteira do humano responsável
        bytes32 agentFingerprint;   // hash biométrico do agente
        uint256 sponsorshipLevel;   // 1-5 (nível de autonomia delegada)
        uint256 createdAt;
        uint256 expiresAt;
        bool isRevoked;
    }

    mapping(bytes32 => AgentBinding) public agents;
    address public registryAdmin;

    event AgentBound(bytes32 indexed agentId, address indexed human, uint256 level);
    event AgentRevoked(bytes32 indexed agentId);

    constructor() {
        registryAdmin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == registryAdmin, "Only admin can perform this action");
        _;
    }

    function bindAgent(
        bytes32 agentId,
        address human,
        bytes32 fingerprint,
        uint256 level
    ) external onlyAdmin {
        require(!agents[agentId].isRevoked, "Agent already bound and not revoked");
        agents[agentId] = AgentBinding(
            human,
            fingerprint,
            level,
            block.timestamp,
            0,
            false
        );
        emit AgentBound(agentId, human, level);
    }

    function revokeAgent(bytes32 agentId) external onlyAdmin {
        agents[agentId].isRevoked = true;
        emit AgentRevoked(agentId);
    }

    function isAuthorized(bytes32 agentId, uint256 requiredLevel) external view returns (bool) {
        AgentBinding memory binding = agents[agentId];
        return !binding.isRevoked && binding.sponsorshipLevel >= requiredLevel;
    }
}
