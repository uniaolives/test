import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated, Dimensions } from 'react-native';
import Svg, { Circle, Line, Text as SvgText } from 'react-native-svg';
import { useArkheStore } from '../store/arkheStore';

const { width, height } = Dimensions.get('window');

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

export const QuantumLayerPanel: React.FC = () => {
  const { quantumDots, nanoParticles, fretEfficiency, initializeQuantumLayer } = useArkheStore();
  const pulseAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    initializeQuantumLayer({ n_qds: 50, n_nanoparticles: 200 });

    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 0,
          duration: 2000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  const scale = (val: number, min: number, max: number) =>
    ((val - min) / (max - min)) * Math.min(width, height) * 0.8 + 50;

  return (
    <View style={styles.container}>
      <Svg width={width} height={height * 0.4}>
        {/* Quantum Dots */}
        {quantumDots.map((qd) => {
          const x = scale(qd.position[0], -100, 100);
          const y = scale(qd.position[1], -100, 100);
          const r = qd.radius_nm * 2;

          return (
            <React.Fragment key={qd.id}>
              <Circle
                cx={x}
                cy={y}
                r={r}
                fill={`hsl(${qd.emission_wavelength_nm}, 80%, 50%)`}
                opacity={0.8}
              />
              <AnimatedCircle
                cx={x}
                cy={y}
                r={r * 2}
                fill="none"
                stroke={`hsl(${qd.emission_wavelength_nm}, 80%, 50%)`}
                strokeWidth={1}
                opacity={pulseAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [0.2, 0.6],
                })}
              />
            </React.Fragment>
          );
        })}

        {/* Nano Particles */}
        {nanoParticles.map((nano) => {
          const x = scale(nano.position[0], -200, 200);
          const y = scale(nano.position[1], -200, 200);

          return (
            <Circle
              key={nano.id}
              cx={x}
              cy={y}
              r={nano.size_nm / 5}
              fill="#4CAF50"
              opacity={0.6}
            />
          );
        })}

        {/* FRET Connections */}
        {quantumDots.slice(0, 10).map((qd, i) => {
          const qdX = scale(qd.position[0], -100, 100);
          const qdY = scale(qd.position[1], -100, 100);
          const nano = nanoParticles[i % nanoParticles.length];
          const nanoX = scale(nano.position[0], -200, 200);
          const nanoY = scale(nano.position[1], -200, 200);

          return (
            <Line
              key={`fret-${i}`}
              x1={qdX}
              y1={qdY}
              x2={nanoX}
              y2={nanoY}
              stroke="#FF9800"
              strokeWidth={2}
              strokeDasharray="5,5"
              opacity={0.5}
            />
          );
        })}

        <SvgText x={20} y={30} fill="white" fontSize={16}>
          Q-BIO Interface: {quantumDots.length} QDs, {nanoParticles.length} Nanoparticles
        </SvgText>
        <SvgText x={20} y={50} fill="white" fontSize={14}>
          FRET Efficiency: {(fretEfficiency * 100).toFixed(1)}%
        </SvgText>
      </Svg>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    margin: 10,
  },
});
