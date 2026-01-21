// ==============================================
// ONTOLOGY SOLIDITY BACKEND v0.3.0
// Compilação para Ethereum EVM com Guards Paradigmáticos
// ==============================================

use crate::ast::*;
use crate::compiler::{CompilerError, CompilerResult};
use crate::utils::selector;
use indoc::indoc;

pub struct SolidityBackend {
    pub optimize: bool,
    pub evm_version: String,
    pub inject_guards: bool,
}

impl SolidityBackend {
    pub fn new(optimize: bool, evm_version: String, inject_guards: bool) -> Self {
        Self { optimize, evm_version, inject_guards }
    }

    pub fn compile(&self, program: &OntologyProgram) -> CompilerResult<CompiledContract> {
        let mut solidity_code = String::new();

        // 1. Cabeçalho SPDX e pragma
        solidity_code.push_str(&self.generate_header());

        // 2. Interface do contrato diplomático (reutilizável)
        solidity_code.push_str(&self.generate_diplomatic_interface());

        // 3. Contrato principal
        solidity_code.push_str("contract OntologyContract is DiplomaticProtocol {\n");

        // 4. State variables (paradigm-aware)
        solidity_code.push_str(&self.generate_state_variables(program));

        // 5. Modifiers de guarda paradigmática
        if self.inject_guards {
            solidity_code.push_str(&self.generate_paradigm_modifiers());
        }

        // 6. Traduzir funções
        for func_def in &program.functions {
            solidity_code.push_str(&self.translate_function(func_def)?);
        }

        // 7. Traduzir agents (ERC-1155 NFTs com comportamento)
        for agent_def in program.agents() {
            solidity_code.push_str(&self.translate_agent(agent_def)?);
        }

        // 8. Traduzir classes (estruturas EVM com dispatch)
        for class_def in program.classes() {
            solidity_code.push_str(&self.translate_class(class_def)?);
        }

        // 9. Fechar contrato
        solidity_code.push_str("}\n");

        let stats = self.calculate_stats(program);

        Ok(CompiledContract {
            target_language: "Solidity 0.8.24".to_string(),
            source_code: solidity_code,
            stats,
        })
    }

    // --- Cabeçalho Solidity ---
    fn generate_header(&self) -> String {
        format!(
            r#"
// SPDX-License-Identifier: ONTOLOGY-BSL-1.0
pragma solidity {};

// Imports de segurança paradigmática
import {{DiplomaticProtocol}} from "onto/diplomacy/DiplomaticProtocol.sol";
import {{ParadigmGuard}} from "onto/diplomacy/ParadigmGuard.sol";
import {{TMRConsensus}} from "onto/consensus/TMRConsensus.sol";
import {{MemorySeal}} from "onto/seal/MemorySeal.sol";
            "#,
            self.evm_version
        )
    }

    // --- Interface diplomática base ---
    fn generate_diplomatic_interface(&self) -> String {
        indoc! { r#"
interface IDiplomaticContract {
    // Evento emitido quando uma transmutação ocorre
    event Transmuted(bytes32 indexed functionHash, Paradigm indexed from, Paradigm indexed to);

    // Verificar violação ontológica
    function checkOntologicalViolations(bytes4 selector) external view returns (bool);

    // Executar com consenso TMR
    function executeWithTmr(address[] calldata nodes, bytes memory callData) external;

    // Obter selo de memória BLAKE3-Δ2
    function getMemorySeal(uint256 blockNumber) external view returns (bytes32);
}
        "# }.to_string()
    }

    fn generate_state_variables(&self, _program: &OntologyProgram) -> String {
        indoc! { r#"
    MemorySeal private _memorySeal;
    mapping(bytes4 => bool) private _transmutedFunctions;
    mailbox private _mailbox;
        "# }.to_string()
    }

    // --- Modifiers de guarda paradigmática ---
    fn generate_paradigm_modifiers(&self) -> String {
        indoc! { r#"
modifier pureGuard() {
    // Revert se qualquer SSTORE for detectado neste call frame
    uint256 writesBefore = _paradigmWrites();
    _;
    require(_paradigmWrites() == writesBefore, "PURE_VIOLATION");
}

modifier mutableGuard(uint256 maxWrites) {
    uint256 writesBefore = _paradigmWrites();
    _;
    uint256 writesAfter = _paradigmWrites();
    require(writesAfter - writesBefore <= maxWrites, "MUTABLE_OVERFLOW");
}

modifier agentGuard(address caller) {
    require(_isActor(caller), "AGENT_ONLY");
    _;
}
        "# }.to_string()
    }

    // --- Tradução de função ---
    fn translate_function(&self, func: &Function) -> Result<String, CompilerError> {
        let mut code = String::new();

        // Determinar visibility
        let visibility = if func.paradigm == Paradigm::Agent {
            "external"
        } else {
            "public"
        };

        // Seleção de modifier baseado no paradigma
        let modifier = match func.paradigm {
            Paradigm::Functional => "pureGuard".to_string(),
            Paradigm::Imperative => {
                // Extrair O(n) da constraint para maxWrites
                let max_writes = func.constraints.iter()
                    .find_map(|c| c.parse_o_n_complexity())
                    .unwrap_or(0);
                format!("mutableGuard({})", max_writes)
            }
            Paradigm::Agent => "agentGuard(msg.sender)".to_string(),
            Paradigm::OO | Paradigm::Substrate => "nonReentrant".to_string(),
            Paradigm::Custom(_) => "".to_string(),
        };

        // Gerar assinatura da função
        let params: Vec<String> = func.params.iter()
            .map(|(name, typ)| Ok(format!("{} {}", self.type_to_solidity(typ)?, name)))
            .collect::<Result<Vec<_>, CompilerError>>()?;

        let return_type = self.type_to_solidity(&func.return_type)?;

        code.push_str(&format!(
            r#"
    function {}({}) {} {} returns ({}) {{
        {}
        // Memory Seal checkpoint
        _memorySeal.checkpoint();

        // Implementação nativa (from AST native_block)
        {}

        // Emitir evento de transmutação se aplicável
        if (_isTransmuted(bytes4(keccak256("{}")))) {{
            emit Transmuted(bytes32(keccak256("{}")), {}, {});
        }}

        return _memorySeal.seal();
    }}
            "#,
            func.name,
            params.join(", "),
            visibility,
            if !modifier.is_empty() { modifier } else { "".to_string() },
            return_type,
            if func.paradigm == Paradigm::Functional { "memory" } else { "storage" },
            func.body.to_solidity()?,
            func.name,
            func.name,
            func.paradigm.to_solidity_enum(),
            func.target_paradigm.as_ref().unwrap_or(&func.paradigm).to_solidity_enum()
        ));

        Ok(code)
    }

    // --- Tradução de Agente → Contrato ERC-1155 + Actor Model ---
    fn translate_agent(&self, agent: &Agent) -> Result<String, CompilerError> {
        let agent_id = selector(&agent.name);

        Ok(format!(
            r#"
    // Agente: {} (ID: {})
    function {}Actor(bytes calldata message) external agentGuard(msg.sender) {{
        // Mailbox pattern
        bytes32 messageId = keccak256(message);
        bool processed = _processMessage(messageId, message);

        if (!processed) {{
            // Armazenar para processamento futuro
            _mailbox.push(messageId, message);
        }}

        emit ActorMessage(agentId: {}, message: messageId);
    }}
            "#,
            agent.name,
            agent_id,
            agent.name,
            agent_id
        ))
    }

    // --- Tradução de Classe (OO) → EVM Struct + Dispatch ---
    fn translate_class(&self, class: &Class) -> Result<String, CompilerError> {
        let mut code = String::new();

        // 1. Definir struct de armazenamento
        let fields = class.fields.iter()
                .map(|f| Ok(format!("{} {};", self.type_to_solidity(&f.field_type)?, f.name)))
                .collect::<Result<Vec<_>, CompilerError>>()?.join("\n        ");

        code.push_str(&format!(
            r#"
    struct {} {{
        {}
    }}
            "#,
            class.name,
            fields
        ));

        // 2. Definir funções de método com dispatch dinâmico
        for method in &class.methods {
            let method_params = method.params.iter()
                    .map(|(n, t)| Ok(format!("{} {}", self.type_to_solidity(t)?, n)))
                    .collect::<Result<Vec<_>, CompilerError>>()?.join(", ");

            code.push_str(&format!(
                r#"
    function {}_{}({}) public returns ({}) {{
        // Dispatch: verificar que msg.sender é instância válida
        require(_isInstanceOf(msg.sender, bytes32("{}")), "OO_VIOLATION");

        // Acesso seguro aos fields via delegatecall com guard
        _ooGuardEnter();
        (bool success, bytes memory result) = msg.sender.delegatecall(
            abi.encodeWithSignature("{}", {})
        );
        require(success, "METHOD_CALL_FAILED");
        _ooGuardExit();

        return abi.decode(result, ({}));
    }}
                "#,
                class.name,
                method.name,
                method_params,
                self.type_to_solidity(&method.return_type)?,
                class.name,
                method.name,
                method.params.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>().join(", "),
                self.type_to_solidity(&method.return_type)?,
            ));
        }

        Ok(code)
    }

    // --- Helpers ---
    fn type_to_solidity(&self, typ: &Type) -> Result<String, CompilerError> {
        match typ {
            Type::Pure(inner) => self.type_to_solidity(inner),
            Type::Mutable(inner) => self.type_to_solidity(inner),
            Type::Agent(inner) => {
                // Agents compilam para address + interface
                let inner_type = self.type_to_solidity(inner)?;
                Ok(format!("address /* Agent<{}> */", inner_type))
            }
            Type::Substrate(inner) => {
                // Substrate types têm representação especial
                Ok(format!("bytes32 /* Substrate<{}> */", self.type_to_solidity(inner)?))
            }
            Type::Object(_) => Ok("address".to_string()), // Objects são endereços de contrato
            Type::Int => Ok("int256".to_string()),
            Type::Float => Ok("int256".to_string()), // Simulação para prototipagem
            Type::Bool => Ok("bool".to_string()),
            Type::String => Ok("string".to_string()),
            Type::Named(name, _) => Ok(name.clone()), // Tipos personalizados do usuário
            Type::Unit => Ok("void".to_string()), // Should not happen in returns directly as Solidity uses empty
            Type::Bytes => Ok("bytes".to_string()),
            Type::Address => Ok("address".to_string()),
            Type::TypeVar(t) => Ok(format!("bytes32 /* {} */", t)), // Generic placeholder
            Type::Function(_, _) => Ok("address /* Function pointer */".to_string()),
        }
    }

    fn calculate_stats(&self, program: &OntologyProgram) -> CompilationStats {
        CompilationStats {
            functions_compiled: program.functions.len(),
            contracts_deployed: program.agents.len() + program.classes.len(),
            transmutations_applied: program.transmutations().len(),
            diplomatic_constraints: program.transmutations().iter()
                .map(|t| t.constraints.len())
                .sum(),
            paradigm_guards_injected: if self.inject_guards {
                program.functions.len() + program.agents.len() + program.classes.len()
            } else { 0 },
            gas_estimate: self.estimate_gas(program),
        }
    }

    fn estimate_gas(&self, program: &OntologyProgram) -> u64 {
        // Estimativa simplificada baseada no número de instruções e guards
        let base_gas = 21000u64;
        let func_cost = (program.functions.len() * 5000) as u64;
        let guard_cost = if self.inject_guards {
            (program.functions.len() * 2000 + program.agents.len() * 1500) as u64
        } else { 0 };

        base_gas + func_cost + guard_cost
    }
}

// Conversão de paradigma para enum Solidity
trait ParadigmSolidity {
    fn to_solidity_enum(&self) -> String;
}

impl ParadigmSolidity for Paradigm {
    fn to_solidity_enum(&self) -> String {
        match self {
            Paradigm::Functional => "Paradigm.Functional".to_string(),
            Paradigm::Imperative => "Paradigm.Imperative".to_string(),
            Paradigm::Agent => "Paradigm.Agent".to_string(),
            Paradigm::Substrate => "Paradigm.Substrate".to_string(),
            Paradigm::OO => "Paradigm.OO".to_string(),
            Paradigm::Custom(s) => format!("Paradigm.Custom('{}')", s),
        }
    }
}

// --- Estruturas de retorno ---
#[derive(Debug, Clone)]
pub struct CompiledContract {
    pub target_language: String,
    pub source_code: String,
    pub stats: CompilationStats,
}

#[derive(Debug, Clone)]
pub struct CompilationStats {
    pub functions_compiled: usize,
    pub contracts_deployed: usize,
    pub transmutations_applied: usize,
    pub diplomatic_constraints: usize,
    pub paradigm_guards_injected: usize,
    pub gas_estimate: u64,
}
