// ucd.rs – Universal Coherence Detection in Rust

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    if den_x == 0.0 || den_y == 0.0 { 0.0 } else { num / (den_x * den_y).sqrt() }
}

fn main() {
    let data = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![5.0, 6.0, 7.0, 8.0],
    ];
    let mut sum_corr = 0.0;
    let mut count = 0;
    for i in 0..data.len() {
        for j in i + 1..data.len() {
            sum_corr += pearson(&data[i], &data[j]).abs();
            count += 1;
        }
    }
    let c = if count > 0 { sum_corr / count as f64 } else { 0.5 };
    let f = 1.0 - c;
    println!("C: {:.4}, F: {:.4}, C+F: {:.4}", c, f, c + f);
}
