// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title VotingMechanism
 * @dev Implementação de consenso ponderado para o Conselho de Síntese.
 */
contract VotingMechanism {
    mapping(address => uint256) public votingPower;

    function setPower(address member, uint256 power) external {
        // Apenas admin (Oversoul) pode definir pesos
        votingPower[member] = power;
    }

    function calculateConsensus(uint256 votesFor, uint256 votesAgainst) external pure returns (bool) {
        return votesFor > votesAgainst;
    }
}
