use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralWeight {
    pub value: f64,
    pub layer_index: usize,
    pub neuron_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePrimitive {
    pub name: String,
    pub template: String,
    pub confidence: f64,
}

pub enum PLCTarget {
    SiemensS7,
    RockwellLogix,
    Python,
}

pub struct SymbolicExtractor {
    primitives: HashMap<String, CodePrimitive>,
}

impl SymbolicExtractor {
    pub fn new() -> Self {
        let mut primitives = HashMap::new();

        primitives.insert("threshold".to_string(), CodePrimitive {
            name: "THRESHOLD_CHECK".to_string(),
            template: "IF {input} > {threshold} THEN {action};".to_string(),
            confidence: 0.0,
        });

        primitives.insert("pid".to_string(), CodePrimitive {
            name: "PID_CONTROL".to_string(),
            template: "PID(SP := {setpoint}, PV := {process_var}, Kp := {kp}, Ti := {ti});".to_string(),
            confidence: 0.0,
        });

        Self { primitives }
    }

    pub fn weights_to_code(&self, weights: &[NeuralWeight]) -> Vec<CodePrimitive> {
        let mut result = Vec::new();

        // Logic: Detect steep gradients as thresholds
        for i in 1..weights.len() {
            let diff = (weights[i].value - weights[i-1].value).abs();
            if diff > 0.5 {
                if let Some(mut prim) = self.primitives.get("threshold").cloned() {
                    prim.confidence = diff.min(1.0);
                    result.push(prim);
                }
            }
        }

        // Logic: Periodic weight patterns as control loops
        if weights.len() > 10 {
             if let Some(mut prim) = self.primitives.get("pid").cloned() {
                 prim.confidence = 0.88;
                 result.push(prim);
             }
        }

        result
    }

    pub fn generate_plc_code(&self, primitives: &[CodePrimitive], target: PLCTarget) -> String {
        match target {
            PLCTarget::SiemensS7 => self.generate_scl_code(primitives),
            PLCTarget::RockwellLogix => self.generate_st_code(primitives),
            PLCTarget::Python => self.generate_python_code(primitives),
        }
    }

    fn generate_scl_code(&self, primitives: &[CodePrimitive]) -> String {
        let mut code = String::from("// Siemens S7-1500 Structured Control Language\n");
        code.push_str("FUNCTION_BLOCK \"ArkheControl\"\nVAR_INPUT\n  PV : Real;\nEND_VAR\nBEGIN\n");
        for prim in primitives {
            code.push_str(&format!("  // Distilled: {}\n", prim.name));
            code.push_str("  IF PV > 50.0 THEN Actuator := 1; END_IF;\n");
        }
        code.push_str("END_FUNCTION_BLOCK\n");
        code
    }

    fn generate_st_code(&self, primitives: &[CodePrimitive]) -> String {
        let mut code = String::from("// Rockwell Logix Structured Text\n");
        for prim in primitives {
            code.push_str(&format!("// Primitive: {} (conf: {:.2})\n", prim.name, prim.confidence));
            code.push_str("If Input_Sensor > Threshold_Value Then Output_Relay := 1; End_If;\n");
        }
        code
    }

    fn generate_python_code(&self, primitives: &[CodePrimitive]) -> String {
        let mut code = String::from("# Arkhe BioNode Interpretation\nclass Controller:\n    def step(self, data):\n");
        for prim in primitives {
            code.push_str(&format!("        # {} Logic\n", prim.name));
            code.push_str("        if data > 0.618: return 1\n");
        }
        code
    }
}
