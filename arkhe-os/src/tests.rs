use crate::intention::parser::parse_intention_block;
use crate::kernel::syscall::{SyscallHandler, SyscallResult};
use arkhe_db::ledger::TeknetLedger;

#[test]
fn test_integrated_flow() {
    let mut sys = SyscallHandler::new(100.0);
    let ledger = TeknetLedger::new("test_ledger.log").unwrap();

    let input = "intention test_task { target: \"2009\" coherence: 0.5 priority: 10 payload: \"data\" }";
    let (_, ast) = parse_intention_block(input).unwrap();

    sys.sys_create_task(&ast.name, ast.coherence, 1, ast.priority);
    let res = sys.sys_tick();

    match res {
        SyscallResult::Success(msg) => assert!(msg.contains("Task")),
        _ => panic!("Expected success"),
    }
}
