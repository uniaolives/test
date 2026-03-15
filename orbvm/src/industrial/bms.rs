pub struct BMSController;

impl BMSController {
    pub fn set_temperature(&self, temp: f64) {
        println!("[BMS] Setting temperature to {}C", temp);
    }
}
