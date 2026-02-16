// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract ArkheLedger {
    // Events
    event StateCrystallized(
        string indexed device,
        uint256 indexed metricsHash,
        uint256 timestamp,
        uint256 coherence
    );

    event QKDSessionEstablished(
        bytes32 indexed sessionId,
        address indexed peer,
        uint256 keyLength,
        uint256 errorRate
    );

    event DroneMissionCompleted(
        string indexed droneId,
        string indexed tumorId,
        uint256 nanoparticlesDelivered,
        uint256 timestamp
    );

    // State variables
    mapping(bytes32 => bool) public stateExists;
    mapping(string => uint256[]) public deviceHistory;
    mapping(address => bool) public authorizedNodes;

    address public owner;
    uint256 public totalStates;

    modifier onlyAuthorized() {
        require(authorizedNodes[msg.sender] || msg.sender == owner, "Not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedNodes[msg.sender] = true;
    }

    function recordState(
        string memory device,
        uint256 metricsHash,
        uint256 coherence
    ) public onlyAuthorized returns (bytes32) {
        bytes32 stateId = keccak256(abi.encodePacked(device, metricsHash, block.timestamp));
        require(!stateExists[stateId], "State already exists");

        stateExists[stateId] = true;
        deviceHistory[device].push(block.timestamp);
        totalStates++;

        emit StateCrystallized(device, metricsHash, block.timestamp, coherence);
        return stateId;
    }

    function recordQKDSession(
        address peer,
        uint256 keyLength,
        uint256 errorRate
    ) public onlyAuthorized returns (bytes32) {
        bytes32 sessionId = keccak256(abi.encodePacked(msg.sender, peer, block.timestamp));

        emit QKDSessionEstablished(sessionId, peer, keyLength, errorRate);
        return sessionId;
    }

    function recordDroneMission(
        string memory droneId,
        string memory tumorId,
        uint256 nanoparticlesDelivered
    ) public onlyAuthorized {
        emit DroneMissionCompleted(droneId, tumorId, nanoparticlesDelivered, block.timestamp);
    }

    function authorizeNode(address node) public onlyAuthorized {
        authorizedNodes[node] = true;
    }

    function getDeviceHistory(string memory device) public view returns (uint256[] memory) {
        return deviceHistory[device];
    }
}
