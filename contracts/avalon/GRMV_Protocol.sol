// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./Interfaces.sol";

contract GRMV_Protocol is IGRMV_Protocol {
    mapping(bytes32 => uint256) public fidelityScores;
    address public validator;

    constructor() {
        validator = msg.sender;
    }

    function setFidelity(bytes32 proofHash, uint256 score) external {
        require(msg.sender == validator, "Apenas validador");
        fidelityScores[proofHash] = score;
    }

    function verifyFidelity(bytes32 proofHash) external view override returns (uint256) {
        return fidelityScores[proofHash];
    }
}
