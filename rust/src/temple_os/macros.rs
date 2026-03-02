#[macro_export]
macro_rules! divine {
    ($($arg:tt)*) => {
        println!("🏛️ {}", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! success {
    ($($arg:tt)*) => {
        println!("✅ {}", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        println!("ℹ️ {}", format!($($arg)*));
    };
}

#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        // println!("  {}", format!($($arg)*));
    };
}
