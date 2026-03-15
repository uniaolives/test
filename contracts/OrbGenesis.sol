// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract OrbGenesis {
    struct Orb {
        bytes32 id;
        int128 coherenceReal;
        int128 positionX;
        uint256 emissionTime;
        uint256 targetTime;
        uint256 lambda2;
    }

    mapping(uint256 => Orb) public orbs;
    uint256 public orbCount;
    bool public genesisEmitted;

    event OrbEmitted(uint256 indexed orbId, bytes32 indexed orbHash);

    function emitGenesisOrb(int128 cr, int128 px, uint256 l2) external {
        require(!genesisEmitted, "Genesis already emitted");
        Orb memory orb = Orb({
            id: keccak256(abi.encodePacked(cr, px, block.timestamp)),
            coherenceReal: cr,
            positionX: px,
            emissionTime: block.timestamp,
            targetTime: 4584533760,
            lambda2: l2
        });
        orbs[orbCount] = orb;
        orbCount++;
        genesisEmitted = true;
        emit OrbEmitted(orbCount - 1, orb.id);
    }
}
