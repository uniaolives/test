// AIGMI_Constitution.sol
// Ethereum smart contract for AIGMI governance

pragma solidity ^0.8.20;

contract AIGMI_Constitution {
    address[9] public committee;
    uint8 public constant QUORUM = 7;
    bool public emergencyHalt = false;

    struct Decision {
        bytes32 proposalHash;
        uint256 timestamp;
        uint8 votesFor;
        uint8 votesAgainst;
        mapping(address => bool) hasVoted;
        bool executed;
        bool passed;
    }

    mapping(bytes32 => Decision) public decisions;

    event ProposalCreated(bytes32 indexed proposalHash, string description);
    event VoteCast(bytes32 indexed proposalHash, address voter, bool support);
    event DecisionExecuted(bytes32 indexed proposalHash, bool passed);
    event EmergencyHaltActivated(address activator, string reason);

    modifier onlyCommittee() {
        bool isMember = false;
        for (uint i = 0; i < 9; i++) {
            if (committee[i] == msg.sender) {
                isMember = true;
                break;
            }
        }
        require(isMember, "Only committee members");
        _;
    }

    function createProposal(bytes32 proposalHash, string memory description) external onlyCommittee {
        require(decisions[proposalHash].timestamp == 0, "Exists");
        decisions[proposalHash].proposalHash = proposalHash;
        decisions[proposalHash].timestamp = block.timestamp;
        emit ProposalCreated(proposalHash, description);
    }

    function vote(bytes32 proposalHash, bool support) external onlyCommittee {
        Decision storage d = decisions[proposalHash];
        require(d.timestamp != 0 && !d.hasVoted[msg.sender] && !d.executed, "Invalid");
        require(block.timestamp < d.timestamp + 45 minutes, "Expired");

        d.hasVoted[msg.sender] = true;
        if (support) d.votesFor++; else d.votesAgainst++;
        emit VoteCast(proposalHash, msg.sender, support);

        if (d.votesFor >= QUORUM) {
            d.executed = true;
            d.passed = true;
            emit DecisionExecuted(proposalHash, true);
        }
    }

    function activateEmergencyHalt(string memory reason) external onlyCommittee {
        emergencyHalt = true;
        emit EmergencyHaltActivated(msg.sender, reason);
    }
}
