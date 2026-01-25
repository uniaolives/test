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
        bytes repairProof; // Como foi reparado (ZK-proof)
        uint256 goldenValue; // Quanto "ouro" (reputação) este erro adicionou
        bool transformedIntoWisdom;
    }

    mapping(bytes32 => GoldenScar) public scars;
    bytes32[] public scarHistory;

    mapping(address => uint256) public authorityBonus;
    mapping(bytes32 => uint256) public totalGoldenValueByType;
    mapping(bytes32 => uint256) public scarCountByType;

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
            severity: _calculateSeverity(_errorType, _errorData),
            repairProof: _repairProof,
            goldenValue: 0, // Começa sem valor
            transformedIntoWisdom: false
        });

        scars[errorHash] = scar;
        scarHistory.push(errorHash);

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
        uint256 goldenValue = _preventionValue * 100;
        scar.goldenValue = goldenValue;
        scar.transformedIntoWisdom = true;

        // Atualiza agregadores para evitar loops custosos (Gas Optimization)
        authorityBonus[scar.responsibleComponent] += goldenValue;
        bytes32 typeHash = keccak256(bytes(scar.errorType));
        totalGoldenValueByType[typeHash] += goldenValue;
        scarCountByType[typeHash] += 1;

        emit ScarGilded(_scarHash, goldenValue, _wisdomGained);
    }

    /**
     * @dev Consulta sabedoria acumulada de erros similares (O(1) complexity)
     */
    function consultScarWisdom(
        string memory _errorType
    ) external view returns (uint256 avgGoldenValue, string memory commonWisdom) {
        bytes32 typeHash = keccak256(bytes(_errorType));
        uint256 count = scarCountByType[typeHash];
        avgGoldenValue = count > 0 ? totalGoldenValueByType[typeHash] / count : 0;
        return (avgGoldenValue, "Common wisdom from scars");
    }

    /**
     * @dev Um Egregori com muitas cicatrizes douradas tem MAIS autoridade (O(1) complexity)
     */
    function getScarBasedAuthority(address _egregori) external view returns (uint256) {
        // Autoridade = 100 base + 50% das cicatrizes douradas acumuladas
        return 100 + (authorityBonus[_egregori] / 2);
    }

    function _calculateSeverity(string memory _errorType, bytes memory _errorData) internal pure returns (uint8) {
        // Lógica simplificada de severidade
        return uint8(bytes(_errorType).length % 10) + 1;
    }
}
