import React, { useEffect, useState } from 'react';
import { View, StyleSheet, TouchableOpacity, Text, Dimensions } from 'react-native';
import Svg, { Circle, Line, Polygon, Text as SvgText, Rect } from 'react-native-svg';
import { useArkheStore } from '../store/arkheStore';

const { width } = Dimensions.get('window');

export const DroneSwarmPanel: React.FC = () => {
  const {
    drones,
    tumorRegions,
    activeMission,
    initializeDroneSwarm,
    executeDroneMission
  } = useArkheStore();
  const [selectedDrone, setSelectedDrone] = useState<string | null>(null);

  useEffect(() => {
    initializeDroneSwarm({ n_drones: 3, area_size_m: 1000 });
  }, []);

  const scale = (val: number, min: number, max: number) =>
    ((val - min) / (max - min)) * width * 0.8 + 50;

  return (
    <View style={styles.container}>
      <Svg width={width} height={300}>
        {/* Connection lines between drones */}
        {drones.map((d1, i) =>
          drones.slice(i + 1).map((d2, j) => {
            const x1 = scale(d1.position[0], -500, 500);
            const y1 = scale(d1.position[1], -500, 500);
            const x2 = scale(d2.position[0], -500, 500);
            const y2 = scale(d2.position[1], -500, 500);

            return (
              <Line
                key={`link-${i}-${j}`}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="#00BCD4"
                strokeWidth={2}
                opacity={0.6}
              />
            );
          })
        )}

        {/* Tumor regions */}
        {tumorRegions.map((tumor) => {
          const cx = scale(tumor.center[0], -500, 500);
          const cy = scale(tumor.center[1], -500, 500);
          const r = tumor.radius_mm * 2;

          return (
            <React.Fragment key={tumor.id}>
              <Circle
                cx={cx}
                cy={cy}
                r={r}
                fill="#F44336"
                opacity={0.3}
              />
              <SvgText x={cx - 30} y={cy} fill="white" fontSize={12}>
                Tumor (EPR: {tumor.epr_enhancement}x)
              </SvgText>
            </React.Fragment>
          );
        })}

        {/* Drones */}
        {drones.map((drone) => {
          const x = scale(drone.position[0], -500, 500);
          const y = scale(drone.position[1], -500, 500);
          const isSelected = selectedDrone === drone.id;
          const isActive = activeMission === drone.id;

          return (
            <React.Fragment key={drone.id}>
              {/* Telemetry range */}
              <Circle
                cx={x}
                cy={y}
                r={drone.telemetry_range_mm / 5}
                fill="none"
                stroke="#4CAF50"
                strokeWidth={1}
                opacity={0.3}
              />

              {/* Drone body */}
              <Polygon
                points={`${x},${y-15} ${x-10},${y+10} ${x+10},${y+10}`}
                fill={isActive ? '#FF9800' : isSelected ? '#2196F3' : '#00BCD4'}
                stroke="white"
                strokeWidth={2}
              />

              {/* Status indicator */}
              <Circle
                cx={x + 12}
                cy={y - 12}
                r={4}
                fill={drone.status === 'idle' ? '#4CAF50' :
                      drone.status === 'navigating' ? '#FF9800' :
                      drone.status === 'injecting' ? '#F44336' : '#9C27B0'}
              />

              {/* Battery level */}
              <Rect x={x - 15} y={y + 15} width={30} height={4} fill="#333" />
              <Rect
                x={x - 15}
                y={y + 15}
                width={30 * (drone.battery_level / 100)}
                height={4}
                fill={drone.battery_level > 20 ? '#4CAF50' : '#F44336'}
              />
            </React.Fragment>
          );
        })}
      </Svg>

      <View style={styles.controls}>
        <Text style={styles.title}>BIO-TECH Drone Control</Text>
        {drones.map((drone) => (
          <TouchableOpacity
            key={drone.id}
            style={[
              styles.droneButton,
              selectedDrone === drone.id && styles.selectedButton,
              activeMission === drone.id && styles.activeButton,
            ]}
            onPress={() => setSelectedDrone(drone.id)}
            disabled={activeMission !== null}
          >
            <Text style={styles.buttonText}>
              {drone.id} - {drone.status} ({drone.battery_level}%)
            </Text>
          </TouchableOpacity>
        ))}

        {selectedDrone && activeMission === null && (
          <TouchableOpacity
            style={styles.missionButton}
            onPress={() => {
              const tumor = tumorRegions[0];
              if (tumor) executeDroneMission(selectedDrone, tumor.id);
            }}
          >
            <Text style={styles.buttonText}>Execute Mission</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    margin: 10,
    padding: 10,
  },
  controls: {
    marginTop: 10,
  },
  title: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  droneButton: {
    backgroundColor: '#16213e',
    padding: 12,
    borderRadius: 8,
    marginVertical: 5,
  },
  selectedButton: {
    backgroundColor: '#2196F3',
  },
  activeButton: {
    backgroundColor: '#FF9800',
  },
  missionButton: {
    backgroundColor: '#4CAF50',
    padding: 15,
    borderRadius: 8,
    marginTop: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 14,
  },
});
