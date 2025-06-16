use super::perceptron::{and_gate, nand_gate, or_gate, xor_gate};
use std::io::{self, Write};

pub fn interactive_mode() {
    println!("感知器门模拟器 (输入0或1)");

    loop {
        print!("请输入 x1 (0 或 1): ");
        io::stdout().flush().unwrap();
        let mut x1 = String::new();
        io::stdin().read_line(&mut x1).unwrap();
        let x1: f64 = x1.trim().parse().unwrap_or(-1.0);

        print!("请输入 x2 (0 或 1): ");
        io::stdout().flush().unwrap();
        let mut x2 = String::new();
        io::stdin().read_line(&mut x2).unwrap();
        let x2: f64 = x2.trim().parse().unwrap_or(-1.0);

        print!("请选择门类型 (and/or/nand/xor/exit): ");
        io::stdout().flush().unwrap();
        let mut gate = String::new();
        io::stdin().read_line(&mut gate).unwrap();
        let gate = gate.trim().to_lowercase();

        let result = match gate.as_str() {
            "and" => Some(and_gate(x1, x2)),
            "or" => Some(or_gate(x1, x2)),
            "nand" => Some(nand_gate(x1, x2)),
            "xor" => Some(xor_gate(x1, x2)),
            "exit" => break,
            _ => None,
        };

        match result {
            Some(v) => println!("{}({}, {}) = {}", gate.to_uppercase(), x1, x2, v),
            None => println!("无效门类型，请重新输入"),
        }

        println!("--------------------------");
    }
}
