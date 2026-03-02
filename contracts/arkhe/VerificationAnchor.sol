// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VerificationAnchor {
    event ContractVerified(address indexed contractAddress, bytes32 proofHash);

    mapping(address => bytes32) public verificationProofs;

    function recordVerification(address _contract, bytes32 _proofHash) external {
        verificationProofs[_contract] = _proofHash;
        emit ContractVerified(_contract, _proofHash);
    }
}
