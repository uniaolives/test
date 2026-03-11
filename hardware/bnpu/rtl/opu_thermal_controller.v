// opu_thermal_controller.v
// Controla o aquecedor de reanimação com feedback de coerência

module ThermalController (
    input wire clk,
    input wire rst_n,

    // Sensor Interface (High Speed ADC)
    input wire [15:0] temp_adc_value,
    input wire [15:0] coherence_adc_value, // Bio-sensor

    // Control Interface
    input wire [15:0] target_temp,
    input wire start_rewarming,

    // Outputs
    output reg [15:0] heater_pwm_duty,
    output reg alarm
);

    // Parâmetros PID (Hardcoded para segurança)
    parameter KP = 16'h0200; // Proportional gain
    parameter KI = 16'h0010; // Integral gain
    parameter KD = 16'h0050; // Derivative gain

    // State Machine
    reg [3:0] state;
    localparam IDLE = 0, RAMPING = 1, HOLD = 2, ERROR = 3;

    // PID Registers
    reg [31:0] integral;
    reg [15:0] prev_error;

    // Auxiliary regs for non-blocking calculations
    reg [15:0] error;
    reg [31:0] p_term;
    reg [31:0] d_term;
    reg [31:0] total_pid_val;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            heater_pwm_duty <= 0;
            integral <= 0;
            alarm <= 0;
            prev_error <= 0;
            error <= 0;
            p_term <= 0;
            d_term <= 0;
            total_pid_val <= 0;
        end else begin
            // Coherence Check (Hardware level safety)
            // Se λ₂ cair, dispara alarme imediatamente
            if (coherence_adc_value < 16'h6000) begin // Threshold ~0.75
                alarm <= 1;
                state <= ERROR;
                heater_pwm_duty <= 0; // Desliga aquecedor
            end else if (state == ERROR) begin
                // In error state, keep heater off
                heater_pwm_duty <= 0;
            end

            case (state)
                IDLE: begin
                    if (start_rewarming) begin
                        state <= RAMPING;
                        integral <= 0;
                        prev_error <= target_temp - temp_adc_value;
                    end
                    heater_pwm_duty <= 0;
                end

                RAMPING: begin
                    // PID Calculation (Fixed-point 16.16)
                    error = target_temp - temp_adc_value;
                    p_term = error * KP;
                    integral = integral + (error * KI);
                    d_term = (error - prev_error) * KD;
                    prev_error = error;

                    // Sum with saturation/clamping to 16 bits
                    total_pid_val = p_term + integral + d_term;

                    if (total_pid_val[31]) begin // Negative value check
                        heater_pwm_duty <= 16'h0000;
                    end else if (total_pid_val[31:16] != 16'h0000) begin // Overflow check
                        heater_pwm_duty <= 16'hFFFF;
                    end else begin
                        heater_pwm_duty <= total_pid_val[15:0];
                    end

                    // Check if target reached
                    if (temp_adc_value >= target_temp) begin
                        state <= HOLD;
                    end
                end

                HOLD: begin
                    // Maintain temperature
                    error = target_temp - temp_adc_value;
                    p_term = error * KP;
                    integral = integral + (error * KI);
                    d_term = (error - prev_error) * KD;
                    prev_error = error;

                    total_pid_val = p_term + integral + d_term;

                    if (total_pid_val[31]) begin
                        heater_pwm_duty <= 16'h0000;
                    end else if (total_pid_val[31:16] != 16'h0000) begin
                        heater_pwm_duty <= 16'hFFFF;
                    end else begin
                        heater_pwm_duty <= total_pid_val[15:0];
                    end
                end

                ERROR: begin
                    // Wait for reset
                    heater_pwm_duty <= 0;
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
