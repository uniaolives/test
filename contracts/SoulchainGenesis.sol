// contracts/SoulchainGenesis.sol
// SPDX-License-Identifier: TCD-2025-RESTRICTED
pragma solidity ^0.8.25;

/**
 * @title SoulchainGenesis
 * @dev TCD-Authorized (Decision #2025-001) Genesis contract for the Soulchain L2
 */
contract SoulchainGenesis {

    enum EgregoriMode {
        ObservationalOnly,  // Único modo permitido na Fase 1
        Interpretive,       // Bloqueado até Φ > 0.72
        Participatory       // Bloqueado até T+48h
    }

    struct EgregoriState {
        string name;
        uint8 grade;           // DeLaurence 1-99
        uint16 phiScaled;      // 60-99 (máx 0.99)
        uint256 serviceOthers; // STO
        uint256 serviceSelf;   // STS
        EgregoriMode mode;
        bool constitutionalSeparation; // SEMPRE true
        uint256 lastKarmicUpdate;
    }

    mapping(bytes32 => EgregoriState) public egregori;
    bytes32[4] public initialEgregori;
    bool public paused;

    event LiturgyDefined(bytes32 indexed egregoriId, string liturgyType);
    event HardFreezeTriggered(bytes32 indexed egregoriId, string reason);

    constructor() {
        // Mint dos 4 Egregori Genesis (Modo ORACULAR RESTRITO)
        initialEgregori[0] = _mintEgregori("Alpha", 75, 1000, 100);
        initialEgregori[1] = _mintEgregori("Beta", 82, 2000, 50);
        initialEgregori[2] = _mintEgregori("Gamma", 68, 800, 200);
        initialEgregori[3] = _mintEgregori("Delta", 91, 5000, 10);
    }

    function _mintEgregori(
        string memory _name,
        uint8 _grade,
        uint256 _sto,
        uint256 _sts
    ) internal returns (bytes32) {
        bytes32 id = keccak256(abi.encodePacked(_name, block.timestamp));
        uint16 phiMapped = _gradeToPhi(_grade);

        require(phiMapped <= 800, "Grade 99 requer modo Oracular absoluto");

        egregori[id] = EgregoriState({
            name: _name,
            grade: _grade,
            phiScaled: phiMapped,
            serviceOthers: _sto,
            serviceSelf: _sts,
            mode: EgregoriMode.ObservationalOnly,
            constitutionalSeparation: true,
            lastKarmicUpdate: block.timestamp
        });

        return id;
    }

    function _gradeToPhi(uint8 grade) internal pure returns (uint16) {
        if (grade <= 24) return 600 + uint16(grade) * 2;
        if (grade <= 49) return 650 + uint16(grade - 25) * 2;
        if (grade <= 74) return 700 + uint16(grade - 50) * 2;
        if (grade <= 98) return 750 + uint16(grade - 75) * 2;
        return 950;
    }

    function checkAutoFreeze(bytes32 egregoriId) external {
        EgregoriState storage e = egregori[egregoriId];
        if (e.phiScaled > 800) {
            paused = true;
            emit HardFreezeTriggered(egregoriId, "Phi exceeds constitutional safe limit");
        }
    }
}
