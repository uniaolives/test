// SPDX-License-Identifier: CC-BY-NC-4.0
pragma solidity ^0.8.19;

import "./Interfaces.sol";

contract DAO_Oracle is IQuadraticReputationDAO {
    struct Verifier {
        uint256 reputation;      // REP tokens staked
        uint256 verifiedCount;   // Milestones successfully verified
        uint256 fraudCount;      // Failed verifications
        uint256 lastActivity;
        bool isActive;
    }

    mapping(address => Verifier) public verifiers;
    address[] public activeVerifiers;
    address public REP_TOKEN_ADDRESS;
    address public consensusAdmin;

    // Parâmetros do sistema
    uint256 public constant MIN_STAKE = 10 ether;
    uint256 public constant VOTE_COST_BASE = 1;

    event VerifierRegistered(address verifier, uint256 stake);
    event VoteCast(address verifier, uint256 milestone, uint256 weight, uint256 cost);
    event FraudDetected(address verifier, uint256 slashedAmount);
    event ReputationEarned(address verifier, uint256 amount);

    modifier onlyConsensus() {
        require(msg.sender == consensusAdmin, "Apenas administrador de consenso");
        _;
    }

    constructor(address _repToken) {
        REP_TOKEN_ADDRESS = _repToken;
        consensusAdmin = msg.sender;
    }

    function registerVerifier(uint256 _initialStake) external {
        require(_initialStake >= MIN_STAKE, "Stake insuficiente");
        require(!verifiers[msg.sender].isActive, "Ja registrado");

        IERC20(REP_TOKEN_ADDRESS).transferFrom(msg.sender, address(this), _initialStake);

        verifiers[msg.sender] = Verifier({
            reputation: _initialStake,
            verifiedCount: 0,
            fraudCount: 0,
            lastActivity: block.timestamp,
            isActive: true
        });

        activeVerifiers.push(msg.sender);
        emit VerifierRegistered(msg.sender, _initialStake);
    }

    function castVote(uint256 _milestoneId, uint256 _weight, bool _approve) external {
        Verifier storage v = verifiers[msg.sender];
        require(v.isActive, "Verificador inativo");

        uint256 voteCost = _weight * _weight * VOTE_COST_BASE;
        require(v.reputation >= voteCost, "Reputacao insuficiente");

        v.reputation -= voteCost;
        v.lastActivity = block.timestamp;
        emit VoteCast(msg.sender, _milestoneId, _weight, voteCost);
    }

    // IQuadraticReputationDAO implementation
    function getConsensusLevel() external view override returns (uint256) {
        // Simplified: return a fixed value or logic based on active verifiers
        // In a real system, this would be the result of a specific vote
        return 85; // 85% consensus for simulation
    }

    function slashVerifier(address _verifier, uint256 _fraudPenalty) external onlyConsensus {
        Verifier storage v = verifiers[_verifier];
        require(v.isActive, "Verificador inativo");

        uint256 slashAmount = (v.reputation * _fraudPenalty) / 100;
        v.reputation -= slashAmount;
        v.fraudCount++;

        if (v.fraudCount >= 3) {
            v.isActive = false;
        }

        emit FraudDetected(_verifier, slashAmount);
    }

    // Scientific Oracle Integration part
    struct ScientificSource {
        string name;
        address oracleAddress;
        uint256 trustScore;
    }

    mapping(string => ScientificSource) public trustedSources;
    IResurrectionTrigger public trigger;

    function setTrigger(address _trigger) external onlyConsensus {
        trigger = IResurrectionTrigger(_trigger);
    }

    function updateResurrectionTrigger(uint256 geneEditing, uint256 organoid, uint256 revival) external onlyConsensus {
        trigger.updateConditions(geneEditing, organoid, revival);
    }
}
