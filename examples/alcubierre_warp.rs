struct SpacetimeCoordinate {
    t: f64, x: f64, y: f64, z: f64,
}

struct AlcubierreDrive {
    velocity: f64,      // Velocidade aparente da bolha (pode ser > c)
    radius: f64,        // Raio da bolha de dobra
    thickness: f64,     // Espessura da parede da bolha
}

impl AlcubierreDrive {
    // Função de forma de Alcubierre f(r_s)
    fn shape_function(&self, r_s: f64) -> f64 {
        let sigma = self.thickness;
        let r = self.radius;
        // Aproximação da função top-hat suave
        let num = (sigma * (r_s + r)).tanh() - (sigma * (r_s - r)).tanh();
        let den = 2.0 * (sigma * r).tanh();
        num / den
    }

    // Calcula a deformação do tensor métrico g_xx em um ponto
    fn calculate_metric_deformation(&self, pos: &SpacetimeCoordinate, ship_x: f64) -> f64 {
        let r_s = ((pos.x - ship_x).powi(2) + pos.y.powi(2) + pos.z.powi(2)).sqrt();
        let f_r = self.shape_function(r_s);

        // Retorna o deslocamento do espaço-tempo (shift vector)
        -self.velocity * f_r
    }
}

fn main() {
    let drive = AlcubierreDrive {
        velocity: 2.0,
        radius: 10.0,
        thickness: 0.5,
    };
    let coord = SpacetimeCoordinate { t: 0.0, x: 5.0, y: 0.0, z: 0.0 };
    let deformation = drive.calculate_metric_deformation(&coord, 0.0);
    println!("Metric deformation at (5,0,0): {}", deformation);
}
