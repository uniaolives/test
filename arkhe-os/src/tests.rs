use std::collections::BTreeMap;
use crate::intention::parser::parse_intention_block;
use crate::kernel::syscall::{SyscallHandler, SyscallResult};
use arkhe_db::ledger::TeknetLedger;

#[test]
fn test_multivariate_analytics_logic() {
    let mut engine = crate::sensors::analytics::MultivariateAnalytics::new(10);
    for _ in 0..10 {
        engine.push("wifi", rand::random::<f64>());
        engine.push("5g", rand::random::<f64>());
    }
    let k = engine.mardia_kurtosis();
    assert!(k != 0.0);
}

#[test]
fn test_transfer_entropy_logic() {
    let mut engine = crate::sensors::entropy::TransferEntropy::new(20);
    for i in 0..20 {
        let x = i as f64 * 0.1;
        let y = x + 0.05;
        engine.push(x, y);
    }
    let te = engine.calculate();
    assert!(te > 0.0);
}

#[test]
fn test_high_res_timestamp_ordering() {
    let mut buffer = BTreeMap::new();
    buffer.insert(1000, "late".to_string());
    buffer.insert(500, "early".to_string());

    let first = buffer.iter().next().unwrap();
    assert_eq!(*first.0, 500);
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
