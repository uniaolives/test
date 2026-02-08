// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
    function balanceOf(address account) external view returns (uint256);
}

interface IQuadraticReputationDAO {
    function getConsensusLevel() external view returns (uint256);
}

interface IResurrectionTrigger {
    function isTechnologyReady(uint256 threshold) external view returns (bool);
    function updateConditions(uint256 geneEditing, uint256 organoid, uint256 revival) external;
}

interface IGRMV_Protocol {
    function verifyFidelity(bytes32 proofHash) external view returns (uint256);
}
