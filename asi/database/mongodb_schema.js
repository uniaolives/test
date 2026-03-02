// asi/database/mongodb_schema.js
// Critical Memory Layer Validator for Arkhe Protocol

db.createCollection("agent_epigenome", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["agent_id", "epigenome"],
            properties: {
                agent_id: { bsonType: "string" },
                epigenome: {
                    bsonType: "object",
                    required: ["z", "markov", "C", "F", "regime"],
                    properties: {
                        z: { bsonType: "double", minimum: 0.0, maximum: 1.0 },
                        markov: { bsonType: "double", minimum: 0.0, maximum: 1.0 },
                        C: { bsonType: "double", minimum: 0.0, maximum: 1.0 },
                        F: { bsonType: "double", minimum: 0.0, maximum: 1.0 },
                        regime: { enum: ["DETERMINISTIC", "CRITICAL", "STOCHASTIC"] }
                    }
                }
            }
        }
    }
});

db.agent_epigenome.createIndex({ "epigenome.regime": 1, "epigenome.z": 1 });
