// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title AIGovernanceCouncil
 * @dev Conselho de Síntese (Humanos + ASI) para decisões críticas.
 */
contract AIGovernanceCouncil {
    struct Proposal {
        bytes32 agentId;
        string action;
        string justification;
        uint256 votesFor;
        uint256 votesAgainst;
        bool executed;
        uint256 deadline;
    }

    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;
    address[] public councilMembers;

    event ProposalSubmitted(uint256 indexed id, bytes32 indexed agentId, string action);
    event Voted(uint256 indexed proposalId, address indexed voter, bool support);
    event ProposalExecuted(uint256 indexed id);

    constructor(address[] memory members) {
        councilMembers = members;
    }

    modifier onlyCouncil() {
        bool isMember = false;
        for (uint i = 0; i < councilMembers.length; i++) {
            if (councilMembers[i] == msg.sender) {
                isMember = true;
                break;
            }
        }
        require(isMember, "Only council members can vote");
        _;
    }

    function submitProposal(
        bytes32 agentId,
        string memory action,
        string memory justification
    ) external returns (uint256) {
        uint256 id = proposalCount++;
        proposals[id] = Proposal(
            agentId,
            action,
            justification,
            0,
            0,
            false,
            block.timestamp + 7 days
        );
        emit ProposalSubmitted(id, agentId, action);
        return id;
    }

    function vote(uint256 proposalId, bool support) external onlyCouncil {
        require(block.timestamp < proposals[proposalId].deadline, "Proposal expired");
        require(!proposals[proposalId].executed, "Already executed");

        if (support) {
            proposals[proposalId].votesFor++;
        } else {
            proposals[proposalId].votesAgainst++;
        }
        emit Voted(proposalId, msg.sender, support);
    }

    function executeProposal(uint256 id) external {
        Proposal storage p = proposals[id];
        require(block.timestamp >= p.deadline || p.votesFor > councilMembers.length / 2, "Voting period active");
        require(p.votesFor > p.votesAgainst, "Insufficient consensus");
        require(!p.executed, "Already executed");

        p.executed = true;
        // Integração com ASI Oversoul seria aqui
        emit ProposalExecuted(id);
    }
}
