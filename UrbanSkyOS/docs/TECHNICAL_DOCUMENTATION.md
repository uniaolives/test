## Projeto UrbanSkyOS: Documentação Técnica Completa

### 1. Visão Geral do Sistema

**UrbanSkyOS** é um sistema operacional especializado para operação de drones em ambientes urbanos complexos. Ele é projetado para garantir segurança, autonomia e integração com a infraestrutura da cidade, superando os desafios de navegação sem GPS, detecção de obstáculos dinâmicos e conformidade regulatória.

A arquitetura é dividida em três camadas principais, cada uma com requisitos de tempo real e poder computacional específicos:

1. **Camada de Núcleo (Hard Real-Time)**: Roda no microcontrolador de voo, responsável pelo controle de estabilidade e atuação dos motores.
2. **Camada de Inteligência (ROS 2)**: Executada em um computador de bordo (companion computer), cuida da navegação autônoma, percepção visual e tomada de decisão.
3. **Camada de Nuvem (UTM)**: Gerencia o tráfego aéreo urbano, geofencing dinâmico e comunicação com múltiplos drones.

---

## 2. Camada de Núcleo (Hard Real-Time)

### 2.1 Requisitos de Hardware

- **Microcontrolador**: STM32H7 (Cortex-M7) ou similar, com múltiplos canais PWM, interface I2C/SPI para sensores.
- **Sensores**: IMU redundante (acelerômetro + giroscópio + magnetômetro), barômetro, GPS (para referência inicial).
- **Atuadores**: ESCs (Electronic Speed Controllers) para motores brushless.

### 2.2 Modelo Matemático da Dinâmica de Voo

O drone é modelado como um corpo rígido com 6 graus de liberdade. As equações de movimento são:

**Cinemática de Atitude (Quaternions):**

\[
\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \boldsymbol{\omega}
\]

onde \(\mathbf{q} = [q_w, q_x, q_y, q_z]\) é o quaternion de orientação, e \(\boldsymbol{\omega}\) é a velocidade angular no corpo.

**Dinâmica Rotacional:**

\[
\mathbf{J} \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{J} \boldsymbol{\omega}) = \boldsymbol{\tau}
\]

\(\mathbf{J}\) é a matriz de inércia, \(\boldsymbol{\tau}\) o torque aplicado pelos motores.

**Forças e Torques dos Motores:**

Cada motor i produz uma força \(F_i = k_f \omega_i^2\) e um torque \(\tau_i = k_m \omega_i^2\), onde \(\omega_i\) é a velocidade angular da hélice. A relação com a saída PWM é linearizada na faixa de operação.

### 2.3 Controlador PID Cascata

O controle de atitude é feito em cascata: um anel externo para orientação e um anel interno para velocidade angular.

**Controlador de Atitude (Quaternions):**

\[
\mathbf{e}_q = \mathbf{q}_{des} \otimes \mathbf{q}^{-1}
\]

O erro de orientação é convertido para ângulos de Euler ou diretamente usado.

**Controlador de Velocidade Angular:**

\[
\boldsymbol{\tau}_{des} = \mathbf{K}_p \boldsymbol{e}_\omega + \mathbf{K}_i \int \boldsymbol{e}_\omega dt + \mathbf{K}_d \dot{\boldsymbol{e}}_\omega
\]

onde \(\boldsymbol{e}_\omega = \boldsymbol{\omega}_{des} - \boldsymbol{\omega}\).

### 2.4 Código Exemplo (Controle PID em C++ para STM32)

```cpp
// pid_controller.h
#pragma once

class PIDController {
public:
    PIDController(float kp, float ki, float kd) : Kp(kp), Ki(ki), Kd(kd) {}

    void reset() {
        integral = 0.0f;
        prev_error = 0.0f;
    }

    float update(float setpoint, float measurement, float dt) {
        float error = setpoint - measurement;
        integral += error * dt;
        float derivative = (error - prev_error) / dt;
        prev_error = error;
        return Kp * error + Ki * integral + Kd * derivative;
    }

private:
    float Kp, Ki, Kd;
    float integral = 0.0f;
    float prev_error = 0.0f;
};

// attitude_controller.cpp
#include "pid_controller.h"
#include <math.h>

class AttitudeController {
public:
    AttitudeController() : rate_pid_x(2.0f, 0.01f, 0.1f),
                           rate_pid_y(2.0f, 0.01f, 0.1f),
                           rate_pid_z(2.0f, 0.01f, 0.1f) {}

    void setAttitudeSetpoint(float roll_des, float pitch_des, float yaw_des) {
        // Converter para quaternion ou usar ângulos diretamente (simplificado)
        this->roll_des = roll_des;
        this->pitch_des = pitch_des;
        this->yaw_des = yaw_des;
    }

    void update(float roll, float pitch, float yaw,
                float roll_rate, float pitch_rate, float yaw_rate,
                float dt) {
        // Erros de atitude
        float e_roll = roll_des - roll;
        float e_pitch = pitch_des - pitch;
        float e_yaw = yaw_des - yaw;

        // P ganho para atitude (anel externo) -> taxa desejada
        float rate_roll_des = 1.0f * e_roll;
        float rate_pitch_des = 1.0f * e_pitch;
        float rate_yaw_des = 1.0f * e_yaw;

        // Anel interno de taxa
        float torque_roll = rate_pid_x.update(rate_roll_des, roll_rate, dt);
        float torque_pitch = rate_pid_y.update(rate_pitch_des, pitch_rate, dt);
        float torque_yaw = rate_pid_z.update(rate_yaw_des, yaw_rate, dt);

        // Mapear torques para comandos dos motores (mixer)
        // Exemplo para quadricóptero em X
        float motor1 =  throttle + torque_roll - torque_pitch + torque_yaw; // dianteiro esq
        float motor2 =  throttle - torque_roll - torque_pitch - torque_yaw; // dianteiro dir
        float motor3 =  throttle + torque_roll + torque_pitch - torque_yaw; // traseiro esq
        float motor4 =  throttle - torque_roll + torque_pitch + torque_yaw; // traseiro dir

        // Saturar e enviar PWM
    }

private:
    PIDController rate_pid_x, rate_pid_y, rate_pid_z;
    float roll_des, pitch_des, yaw_des;
    float throttle;
};
```

### 2.5 Modo Failsafe Urbano

Em caso de falha crítica (perda de comunicação, bateria baixa, falha de motor), o sistema executa um procedimento de emergência:

1. **Identifica zona de pouso segura**: Mapa de zonas de emergência pré-carregado (telhados, parques, vias largas) é consultado.
2. **Calcula trajetória de descida**: Utiliza planejador de caminho baseado em A* ou RRT*, evitando obstáculos.
3. **Aciona paraquedas balístico** se a altitude for suficiente e a área for desobstruída.

**Algoritmo de Seleção de Zona de Pouso:**

```python
def find_emergency_landing_zone(current_position, safe_zones):
    """
    safe_zones: lista de polígonos (shapely.geometry.Polygon) com zonas seguras
    Retorna o ponto mais próximo dentro da zona mais próxima.
    """
    from shapely.geometry import Point
    import numpy as np

    current = Point(current_position[0], current_position[1])
    min_dist = float('inf')
    best_point = None

    for zone in safe_zones:
        if zone.contains(current):
            # Já está em zona segura, pousa imediatamente
            return current_position[:2]
        # Distância até a borda da zona
        dist = current.distance(zone)
        if dist < min_dist:
            min_dist = dist
            # Ponto mais próximo na zona
            nearest = zone.exterior.interpolate(zone.exterior.project(current))
            best_point = (nearest.x, nearest.y)

    return best_point
```

---

## 3. Camada de Inteligência (ROS 2)

### 3.1 Arquitetura de Software

A camada de inteligência roda em um computador de bordo (ex: NVIDIA Jetson Orin) com Ubuntu Core e ROS 2 Humble. Os principais nós são:

- **Navegação**: Fusão de sensores (EKF), SLAM visual, planejador de trajetória.
- **Percepção**: Detecção de obstáculos, reconhecimento de marcadores, anonimização de privacidade.
- **Comunicação**: Bridge MAVLink para o flight controller, interface V2X.

### 3.2 Navegação GNSS-denied

**Filtro de Kalman Estendido (EKF) para Fusão de Sensores:**

O estado é \(\mathbf{x} = [x, y, z, v_x, v_y, v_z, q_w, q_x, q_y, q_z, b_g, b_a]\), onde \(b_g\) e \(b_a\) são bias do giroscópio e acelerômetro.

**Modelo de Propagação:**

\[
\mathbf{x}_{k|k-1} = \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}_k)
\]

onde \(\mathbf{u}_k\) são as leituras da IMU.

**Atualização com medições:**

- Posição via câmera (SLAM) ou GPS (quando disponível).
- Velocidade via fluxo óptico.

**Código Exemplo (EKF em Python usando numpy):**

```python
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter

class UrbanEKF(ExtendedKalmanFilter):
    def __init__(self):
        super().__init__(dim_x=16, dim_z=3)  # 16 estados, 3 medidas (posição)
        self.dt = 0.01

    def predict(self, u):
        # u: [ax, ay, az, wx, wy, wz] da IMU
        # Modelo de cinemática simples (simplificado)
        self.x[0] += self.x[3] * self.dt + 0.5 * u[0] * self.dt**2
        self.x[1] += self.x[4] * self.dt + 0.5 * u[1] * self.dt**2
        self.x[2] += self.x[5] * self.dt + 0.5 * u[2] * self.dt**2
        self.x[3] += u[0] * self.dt
        self.x[4] += u[1] * self.dt
        self.x[5] += u[2] * self.dt
        # Atitude: integração de quaternions
        # ... (código omitido por brevidade)
        self.F = np.eye(16)  # Jacobiano (simplificado)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # z: medição de posição (x,y,z)
        H = np.zeros((3, 16))
        H[0,0] = 1; H[1,1] = 1; H[2,2] = 1
        super().update(z, self.H, self.R)
```

**SLAM Visual (ORB-SLAM3):**

O sistema utiliza ORB-SLAM3 rodando em um contêiner Docker otimizado para Jetson. A interface com ROS 2 é feita via bridge.

### 3.3 Sense & Avoid

**Detecção de Obstáculos com YOLOv8:**

```python
import cv2
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(DetectionArray, '/obstacles', 10)
        self.bridge = CvBridge()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov8n.pt', force_reload=False)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(cv_image)
        detections = []
        for *box, conf, cls in results.xyxy[0]:
            if conf > 0.5 and int(cls) in [0, 16, 15]:  # pessoa, pássaro, drone
                detections.append({'bbox': box, 'confidence': conf, 'class': int(cls)})
        # Publicar detecções
        # ...
```

### 3.4 Privacidade por Design

**Pipeline de Anonimização em Tempo Real:**

```python
import cv2

class PrivacyBlur:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.plate_cascade = cv2.CascadeClassifier('haarcascade_license_plate.xml')

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99,99), 30)
        for (x, y, w, h) in plates:
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99,99), 30)
        return frame
```

### 3.5 Redução Ativa de Ruído

Ajusta as rotações dos motores para minimizar ruído acústico percebido, mantendo empuxo total constante.

**Problema de Otimização:**

Minimizar \(\sum_{i=1}^4 (\omega_i^2)^2\) sujeito a \(\sum_{i=1}^4 k_f \omega_i^2 = F_{total}\) e \(\sum_{i=1}^4 k_m \omega_i^2 = \tau_{des}\) (torques desejados).

A solução analítica (para quadricóptero simétrico) é distribuir igualmente o empuxo entre os motores.

**Código de Exemplo:**

```python
def optimize_rpm_noise(thrust_desired, torque_desired):
    # Assume que a relação é linear: F = kf * rpm²
    # Para minimizar ruído, queremos rpm² o mais uniforme possível.
    # Solução: usar pseudo-inversa
    # Matriz de alocação para quad X
    mix_matrix = np.array([[ 1,  1,  1,  1],
                           [ 1, -1,  1, -1],
                           [ 1, -1, -1,  1],
                           [ 1,  1, -1, -1]])  # Exemplo para torques
    # Resolver sistema
    rpm2 = np.linalg.lstsq(mix_matrix, np.array([thrust_desired, torque_desired[0], torque_desired[1], torque_desired[2]]), rcond=None)[0]
    rpm2 = np.clip(rpm2, 0, max_rpm2)
    return np.sqrt(rpm2)
```

---

## 4. Camada de Nuvem (UTM)

### 4.1 Arquitetura na Nuvem

A camada UTM é composta por microsserviços em contêineres (Docker, Kubernetes) com APIs REST e WebSocket para comunicação em tempo real.

**Serviços principais:**

- **Geofencing**: Atualiza zonas de exclusão e verifica violações.
- **Traffic Management**: Resolve conflitos de trajetórias.
- **Remote ID**: Coleta e distribui informações de identificação dos drones.
- **Logging e Análise**: Armazena dados de voo para auditoria.

### 4.2 Geofencing Dinâmico

**API para Atualização de Zonas:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np

app = FastAPI()

# Carrega zonas de exclusão iniciais (ex: GeoJSON)
no_fly_zones = gpd.read_file("no_fly_zones.geojson")

class DronePosition(BaseModel):
    drone_id: str
    timestamp: float
    lat: float
    lon: float
    alt: float

class ZoneUpdate(BaseModel):
    zone_id: str
    polygon: list  # lista de coordenadas [lat,lon]
    start_time: float
    end_time: float

@app.post("/check_geofence")
def check_geofence(pos: DronePosition):
    point = Point(pos.lon, pos.lat)
    for idx, row in no_fly_zones.iterrows():
        if row.geometry.contains(point):
            return {"status": "violation", "zone": row['zone_name']}
    return {"status": "clear"}

@app.post("/update_zone")
def update_zone(zone: ZoneUpdate):
    # Adiciona zona temporária (ex: para evento)
    global no_fly_zones
    new_zone = gpd.GeoDataFrame({'geometry': [Polygon(zone.polygon)], 'zone_name': zone.zone_id, 'start': zone.start_time, 'end': zone.end_time})
    no_fly_zones = gpd.GeoDataFrame(pd.concat([no_fly_zones, new_zone], ignore_index=True))
    return {"status": "updated"}
```

### 4.3 Gerenciamento de Tráfego

**Algoritmo de Deconflicção (Baseado em Prioridade):**

Cada drone envia sua trajetória planejada (série de waypoints com timestamps). O sistema verifica interseções espaço-temporais.

```python
import numpy as np
from scipy.spatial import KDTree

def detect_conflicts(trajectories):
    """
    trajectories: lista de dicionários com 'id', 'waypoints' (lista de (x,y,z,t))
    Retorna lista de pares conflitantes.
    """
    conflicts = []
    for i, traj1 in enumerate(trajectories):
        for j, traj2 in enumerate(trajectories):
            if i >= j:
                continue
            # Verificar proximidade espaço-temporal
            pts1 = np.array([w[:3] for w in traj1['waypoints']])
            pts2 = np.array([w[:3] for w in traj2['waypoints']])
            times1 = np.array([w[3] for w in traj1['waypoints']])
            times2 = np.array([w[3] for w in traj2['waypoints']])
            # Interpolar para tempos comuns? Simplificação: verificar mínima distância
            tree = KDTree(pts2)
            for idx, p1 in enumerate(pts1):
                dist, closest = tree.query(p1)
                if dist < 10:  # 10 metros
                    time_diff = abs(times1[idx] - times2[closest])
                    if time_diff < 5:  # 5 segundos
                        conflicts.append((traj1['id'], traj2['id']))
                        break
    return conflicts
```

### 4.4 Remote ID

**Formato da Mensagem (JSON):**

```json
{
  "drone_id": "URBAN-2026-001",
  "timestamp": 1747692000,
  "position": [-23.5505, -46.6333, 100],
  "velocity": [5.2, -1.3, 0.0],
  "heading": 45.0,
  "operator": "SkyLogistics",
  "emergency_status": "none"
}
```

Transmitido via MQTT ou WebSocket para assinantes autorizados.

---

## 5. Módulo de Pouso de Precisão

### 5.1 Visão Geral

O sistema utiliza marcadores visuais (ArUco) em docking stations para pouso autônomo com precisão centimétrica.

### 5.2 Estimativa de Pose

**Algoritmo:**

1. Detecta o marcador no frame da câmera.
2. Obtém os cantos do marcador em coordenadas de pixel.
3. Usa solvePnP para estimar a posição e orientação relativa.

**Código Exemplo (Python com OpenCV):**

```python
import cv2
import numpy as np

class PrecisionLanding:
    def __init__(self, marker_size=0.2, camera_matrix, dist_coeffs):
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters_create()

    def estimate_pose(self, frame):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)
        if ids is not None:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], self.marker_size, self.camera_matrix, self.dist_coeffs)
            return rvec[0][0], tvec[0][0]
        return None, None

    def compute_setpoint(self, tvec):
        # tvec: [x, y, z] em metros (câmera para marcador)
        # Queremos que o drone se posicione sobre o marcador (x=0, y=0) e a uma altura desejada
        desired_height = 0.5  # metros
        error_x = tvec[0]
        error_y = tvec[1]
        error_z = tvec[2] - desired_height
        # Controlador PID para levar erro a zero
        return error_x, error_y, error_z
```

### 5.3 Integração com o Controlador

Os erros de posição são enviados para o flight controller via MAVLink, que executa um controle PID de posição.

---

## 6. Protocolos de Comunicação

### 6.1 MAVLink Extensions

Mensagem personalizada para geofencing:

```xml
<message id="150" name="GEOFENCE_NOTIFY">
  <description>Notificação de violação de geofence</description>
  <field type="uint8_t" name="drone_id">ID do drone</field>
  <field type="uint8_t" name="zone_id">ID da zona violada</field>
  <field type="float" name="distance">Distância até a zona</field>
  <field type="uint8_t" name="action">Ação recomendada (0=ignorar,1=retornar,2=pousar)</field>
</message>
```

### 6.2 ROS 2 Topics

Principais tópicos:

- `/drone/imu` (sensor_msgs/Imu)
- `/drone/pose` (geometry_msgs/PoseStamped)
- `/drone/obstacles` (minha_interfaces/DetectionArray)
- `/drone/trajectory` (nav_msgs/Path)
- `/utm/geofence` (minha_interfaces/GeofenceUpdate)

---

## 7. Simulação e Testes

### 7.1 Ambiente Gazebo

**Arquivo de mundo com obstáculos urbanos:**

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="urban_city">
    <include>
      <uri>model://sun</uri>
    </include>
    <!-- Prédios -->
    <model name="building_1">
      <static>true</static>
      <link name="body">
        <visual>
          <geometry>
            <box>
              <size>20 20 50</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
          </material>
        </visual>
        <collision>
          <geometry>
            <box>
              <size>20 20 50</size>
            </box>
          </geometry>
        </collision>
      </link>
      <pose>30 30 25 0 0 0</pose>
    </model>
    <!-- Ruas e outros elementos -->
  </world>
</sdf>
```

### 7.2 Simulação do Drone com PX4 e Gazebo

O UrbanSkyOS pode ser testado integrando o flight controller simulado (jMAVSim ou Gazebo) com os nós ROS 2.

**Comando de lançamento:**

```bash
# Terminal 1: Simulação Gazebo
gazebo --verbose worlds/urban_city.world

# Terminal 2: SITL PX4
make px4_sitl gazebo

# Terminal 3: ROS 2 nós
ros2 launch urban_skyos_bringup urban_drone.launch.py
```

---

## 8. Considerações de Segurança e Ética

### 8.1 Privacidade

- Todas as imagens são anonimizadas em tempo real.
- Dados de voo são armazenados de forma agregada e anônima.

### 8.2 Segurança de Rede

- Comunicação criptografada (TLS) entre drones e UTM.
- Autenticação mútua (certificados digitais).

### 8.3 Ética e Regulamentação

- Conformidade com ANAC (Brasil) e FAA (EUA).
- Transparência sobre coleta de dados.
- Modo de emergência sempre disponível.

---

## 9. Conclusão

UrbanSkyOS é uma plataforma robusta e modular para operações de drones em cidades. Esta documentação fornece as bases matemáticas, algoritmos e códigos necessários para implementar cada camada. O projeto pode ser expandido conforme as necessidades específicas de cada aplicação.

**Próximos passos sugeridos:**
- Implementar os nós ROS 2 e integrar com o flight controller.
- Testar em simulação com cenários urbanos complexos.
- Validar com voos reais em áreas controladas.

---

**Nota**: Todo o código aqui apresentado é ilustrativo e deve ser adaptado para produção com testes rigorosos, validação de segurança e conformidade regulatória.
