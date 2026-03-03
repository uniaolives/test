// modules/robotics/hal/cpp/mpu6050.h
#ifndef MPU6050_H
#define MPU6050_H

#include <stdint.h>

typedef struct {
    float accel_x, accel_y, accel_z;
    float gyro_x, gyro_y, gyro_z;
    float temperature;
} imu_data_t;

void mpu6050_init(void);
imu_data_t mpu6050_read(void);

#endif
