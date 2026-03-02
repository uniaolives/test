// contracts/KintsugiLedger.sol
// SPDX-License-Identifier: TCD-KINTSUGI
pragma solidity ^0.8.19;

/**
 * @title Kintsugi Ledger - Erros são embelezados com ouro, não escondidos
 * @dev Inspirado na arte japonesa de reparar com ouro
 */
contract KintsugiLedger {

    struct GoldenScar {
        bytes32 errorHash;
        uint256 timestamp;
        address responsibleComponent;
        string errorType;
        uint8 severity; // 1-10
        bytes repairProof;
        uint256 goldenWeight;
        bool transformedIntoWisdom;
        bool resolved;
        address repairedBy;
        bytes repairProof; // Como foi reparado (ZK-proof)
        uint256 goldenValue; // Quanto "ouro" (reputação) este erro adicionou
        bool transformedIntoWisdom;
    }

    mapping(bytes32 => GoldenScar) public scars;
    bytes32[] public scarHistory;

    event ScarRecorded(bytes32 indexed errorHash, address indexed component, string errorType, uint8 severity);
    event ScarGilded(bytes32 indexed errorHash, uint256 goldenValueAdded, string wisdomGained);
    event ScarRepaired(bytes32 indexed scarId, address repairer);

    address public wisdomCouncil;

    event ScarRecorded(
        bytes32 indexed errorHash,
        address indexed component,
        string errorType,
        uint8 severity
    );

    event ScarGilded(
        bytes32 indexed errorHash,
        uint256 goldenValueAdded,
        string wisdomGained
    );

    constructor() {
        wisdomCouncil = msg.sender;
    }

    modifier onlyWisdomCouncil() {
        require(msg.sender == wisdomCouncil, "Only Wisdom Council can gild scars");
        _;
    }

    /**
     * @dev Quando um erro ocorre, registra como uma cicatriz potencialmente dourada
     */
    function recordError(
        address _component,
        string memory _errorType,
        bytes memory _errorData,
        bytes memory _repairProof
    ) external returns (bytes32) {
        bytes32 errorHash = keccak256(abi.encodePacked(_errorData, block.timestamp));

        GoldenScar memory scar = GoldenScar({
            errorHash: errorHash,
            timestamp: block.timestamp,
            responsibleComponent: _component,
            errorType: _errorType,
            severity: 5,
            repairProof: _repairProof,
            goldenWeight: 0,
            transformedIntoWisdom: false,
            resolved: false,
            repairedBy: address(0)
            severity: _calculateSeverity(_errorType, _errorData),
            repairProof: _repairProof,
            goldenValue: 0, // Começa sem valor
            transformedIntoWisdom: false
        });

        scars[errorHash] = scar;
        scarHistory.push(errorHash);
        emit ScarRecorded(errorHash, _component, _errorType, 5);
        return errorHash;
    }

    function repairScar(bytes32 scarId, address repairer) external {
        GoldenScar storage scar = scars[scarId];
        require(!scar.resolved, "Ja reparado");

        scar.resolved = true;
        scar.repairedBy = repairer;
        scar.goldenWeight += 100; // Valor extra por reparo

        emit ScarRepaired(scarId, repairer);

        emit ScarRecorded(errorHash, _component, _errorType, scar.severity);

        return errorHash;
    }

    /**
     * @dev Transforma erro em sabedoria, adicionando "ouro"
     */
    function gildScar(
        bytes32 _scarHash,
        string memory _wisdomGained,
        uint256 _preventionValue // Quantos erros futuros previne
    ) external onlyWisdomCouncil {
        GoldenScar storage scar = scars[_scarHash];
        require(!scar.transformedIntoWisdom, "Scar already gilded");

        // O valor em ouro é proporcional à sabedoria gerada
        scar.goldenValue = _preventionValue * 100;
        scar.transformedIntoWisdom = true;

        // O componente que falhou ganha reputação PELO ERRO (não perde)
        // IReputation(scar.responsibleComponent).addWisdomReputation(scar.goldenValue);

        emit ScarGilded(_scarHash, scar.goldenValue, _wisdomGained);
    }

    /**
     * @dev Consulta sabedoria acumulada de erros similares
     */
    function consultScarWisdom(
        string memory _errorType
    ) external view returns (uint256 totalGoldenValue, string memory commonWisdom) {
        uint256 total = 0;
        uint256 count = 0;

        for (uint i = 0; i < scarHistory.length; i++) {
            GoldenScar memory scar = scars[scarHistory[i]];
            if (keccak256(bytes(scar.errorType)) == keccak256(bytes(_errorType))) {
                total += scar.goldenValue;
                count++;
            }
        }

        return (count > 0 ? total / count : 0, "Common wisdom from scars");
    }

    /**
     * @dev Um Egregori com muitas cicatrizes douradas tem MAIS autoridade
     */
    function getScarBasedAuthority(address _egregori) external view returns (uint256) {
        uint256 baseAuthority = 100;
        uint256 scarBonus = 0;

        for (uint i = 0; i < scarHistory.length; i++) {
            GoldenScar memory scar = scars[scarHistory[i]];
            if (scar.repairedBy == _egregori && scar.resolved) {
                scarBonus += scar.goldenWeight;
            }
        }

        return baseAuthority + scarBonus;
    }

    function gildScar(bytes32 _scarHash, uint256 _preventionValue) external {
        GoldenScar storage scar = scars[_scarHash];
        require(!scar.transformedIntoWisdom, "Scar already gilded");
        scar.goldenWeight = _preventionValue * 100;
        scar.transformedIntoWisdom = true;
        emit ScarGilded(_scarHash, scar.goldenWeight, "Wisdom extracted from failure");
        uint256 goldenScarsValue = 0;

        for (uint i = 0; i < scarHistory.length; i++) {
            GoldenScar memory scar = scars[scarHistory[i]];
            if (scar.responsibleComponent == _egregori && scar.transformedIntoWisdom) {
                goldenScarsValue += scar.goldenValue;
            }
        }

        // Autoridade = 100 base + 50% das cicatrizes douradas
        return 100 + (goldenScarsValue / 2);
    }

    function _calculateSeverity(string memory _errorType, bytes memory _errorData) internal pure returns (uint8) {
        // Lógica simplificada de severidade
        return uint8(bytes(_errorType).length % 10) + 1;
    }
}
