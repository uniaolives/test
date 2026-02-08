// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./Interfaces.sol";

contract ResurrectionTrigger is IResurrectionTrigger {
    uint256 public geneEditingReadiness;
    uint256 public organoidReadiness;
    uint256 public revivalReadiness;

    address public dao;

    modifier onlyDAO() {
        require(msg.sender == dao, "Apenas DAO");
        _;
    }

    constructor(address _dao) {
        dao = _dao;
    }

    function updateConditions(uint256 geneEditing, uint256 organoid, uint256 revival) external override onlyDAO {
        geneEditingReadiness = geneEditing;
        organoidReadiness = organoid;
        revivalReadiness = revival;
    }

    function isTechnologyReady(uint256 threshold) external view override returns (bool) {
        return (geneEditingReadiness >= threshold &&
                organoidReadiness >= threshold &&
                revivalReadiness >= threshold);
    }
}
