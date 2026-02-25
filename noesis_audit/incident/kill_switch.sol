// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title AgentKillSwitch
 * @dev Interruptor de emergência para agentes autônomos.
 */
contract AgentKillSwitch {
    mapping(bytes32 => bool) public isKilled;
    address public safetyAdmin;

    event AgentKilled(bytes32 indexed agentId, string reason, address triggeredBy);
    event AgentRevived(bytes32 indexed agentId);

    constructor() {
        safetyAdmin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == safetyAdmin, "Only safety admin can trigger kill switches");
        _;
    }

    function killAgent(bytes32 agentId, string memory reason) external onlyAdmin {
        isKilled[agentId] = true;
        emit AgentKilled(agentId, reason, msg.sender);
    }

    function reviveAgent(bytes32 agentId) external onlyAdmin {
        isKilled[agentId] = false;
        emit AgentRevived(agentId);
    }

    function checkStatus(bytes32 agentId) external view returns (bool) {
        return isKilled[agentId];
    }
}
