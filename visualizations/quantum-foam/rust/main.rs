// Rust version using Bevy game engine
use bevy::prelude::*;
use rand::Rng;

const SCREEN_WIDTH: f32 = 1400.0;
const SCREEN_HEIGHT: f32 = 900.0;
const CHART_WIDTH: f32 = 400.0;
const CHART_HEIGHT: f32 = 300.0;

#[derive(Component)]
struct ParticleData {
    time: f32,
    particles: f32,
    cumulative: f32,
}

#[derive(Component)]
struct VisualizationPanel;

#[derive(Resource)]
struct SimulationData {
    timeline: Vec<ParticleData>,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Quantum Foam Meditation Simulation".into(),
                resolution: (SCREEN_WIDTH, SCREEN_HEIGHT).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::rgb(0.96, 0.96, 0.86))) // Beige background
        .insert_resource(generate_simulation_data())
        .add_systems(Startup, setup_visualizations)
        .run();
}

fn generate_simulation_data() -> SimulationData {
    let mut rng = rand::thread_rng();
    let mut timeline = Vec::new();
    let mut cumulative = 0.0;

    for i in 0..144 {
        let time = i as f32;
        let particles = 50.0 + 10.0 * (time * 0.1).sin() + (rng.gen::<f32>() - 0.5) * 6.0;
        cumulative += particles;

        timeline.push(ParticleData {
            time,
            particles,
            cumulative,
        });
    }

    SimulationData { timeline }
}

fn setup_visualizations(
    mut commands: Commands,
    sim_data: Res<SimulationData>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // Spawn camera
    commands.spawn(Camera2dBundle::default());

    // Panel 1: Consciousness Field
    commands.spawn((
        VisualizationPanel,
        NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                left: Val::Px(50.0),
                top: Val::Px(50.0),
                width: Val::Px(CHART_WIDTH),
                height: Val::Px(CHART_HEIGHT),
                ..default()
            },
            background_color: BackgroundColor(Color::rgba(1.0, 0.98, 0.88, 1.0)),
            ..default()
        },
    )).with_children(|parent| {
        // Add concentric circles for consciousness field
        for i in (1..=5).rev() {
            let radius = (CHART_WIDTH / 2.0) * (i as f32 / 5.0);
            parent.spawn(MaterialMesh2dBundle {
                mesh: meshes.add(shape::Circle::new(radius).into()).into(),
                material: materials.add(ColorMaterial::from(Color::rgba(1.0, 0.84, 0.0, (6.0 - i as f32) / 25.0))),
                transform: Transform::from_xyz(CHART_WIDTH / 2.0, CHART_HEIGHT / 2.0, 0.0),
                ..default()
            });
        }

        // Title
        parent.spawn(TextBundle::from_section(
            "Consciousness Field",
            TextStyle {
                font_size: 16.0,
                color: Color::BLACK,
                ..default()
            },
        ).with_text_alignment(TextAlignment::Center)
          .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(-30.0),
            width: Val::Percent(100.0),
            ..default()
        }));
    });

    // Panel 2: Timeline Chart
    commands.spawn((
        VisualizationPanel,
        NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                left: Val::Px(500.0),
                top: Val::Px(50.0),
                width: Val::Px(CHART_WIDTH),
                height: Val::Px(CHART_HEIGHT),
                ..default()
            },
            background_color: BackgroundColor(Color::rgba(1.0, 0.98, 0.88, 1.0)),
            ..default()
        },
    )).with_children(|parent| {
        // Draw timeline line
        if let Some(first) = sim_data.timeline.first() {
            let mut prev_point = Vec2::new(
                0.0,
                CHART_HEIGHT - (first.particles / 100.0) * CHART_HEIGHT,
            );

            for data in &sim_data.timeline {
                let point = Vec2::new(
                    (data.time / 144.0) * CHART_WIDTH,
                    CHART_HEIGHT - (data.particles / 100.0) * CHART_HEIGHT,
                );

                // Draw line segment
                parent.spawn(MaterialMesh2dBundle {
                    mesh: meshes.add(
                        shape::Quad::new(Vec2::new(
                            (point - prev_point).length(),
                            2.0,
                        )).into(),
                    ).into(),
                    material: materials.add(ColorMaterial::from(Color::rgb(0.83, 0.69, 0.22))),
                    transform: Transform::from_xyz(
                        (prev_point.x + point.x) / 2.0,
                        (prev_point.y + point.y) / 2.0,
                        0.0,
                    ).with_rotation(Quat::from_rotation_z(
                        (point - prev_point).angle_between(Vec2::X)
                    )),
                    ..default()
                });

                prev_point = point;
            }
        }

        // Title
        parent.spawn(TextBundle::from_section(
            "Manifestation Timeline",
            TextStyle {
                font_size: 16.0,
                color: Color::BLACK,
                ..default()
            },
        ).with_text_alignment(TextAlignment::Center)
          .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(-30.0),
            width: Val::Percent(100.0),
            ..default()
        }));
    });

    // Panel 3: Cumulative Chart
    commands.spawn((
        VisualizationPanel,
        NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                left: Val::Px(950.0),
                top: Val::Px(50.0),
                width: Val::Px(CHART_WIDTH),
                height: Val::Px(CHART_HEIGHT),
                ..default()
            },
            background_color: BackgroundColor(Color::rgba(1.0, 0.98, 0.88, 1.0)),
            ..default()
        },
    )).with_children(|parent| {
        // Title
        parent.spawn(TextBundle::from_section(
            "Cumulative Reality",
            TextStyle {
                font_size: 16.0,
                color: Color::BLACK,
                ..default()
            },
        ).with_text_alignment(TextAlignment::Center)
          .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(-30.0),
            width: Val::Percent(100.0),
            ..default()
        }));
    });

    // Panel 4: Quantum Foam
    commands.spawn((
        VisualizationPanel,
        NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                left: Val::Px(50.0),
                top: Val::Px(400.0),
                width: Val::Px(CHART_WIDTH),
                height: Val::Px(CHART_HEIGHT),
                ..default()
            },
            background_color: BackgroundColor(Color::rgb(0.08, 0.0, 0.16)), // Dark purple
            ..default()
        },
    )).with_children(|parent| {
        let mut rng = rand::thread_rng();

        // Quantum foam particles
        for _ in 0..1000 {
            let x = rng.gen::<f32>() * CHART_WIDTH;
            let y = rng.gen::<f32>() * CHART_HEIGHT;
            let radius = rng.gen::<f32>() * 2.0 + 0.5;

            parent.spawn(MaterialMesh2dBundle {
                mesh: meshes.add(shape::Circle::new(radius).into()).into(),
                material: materials.add(ColorMaterial::from(Color::rgba(0.5, 0.0, 0.5, 0.1))),
                transform: Transform::from_xyz(x, y, 0.0),
                ..default()
            });
        }

        // Consciousness overlay
        parent.spawn(MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(100.0).into()).into(),
            material: materials.add(ColorMaterial::from(Color::rgba(1.0, 0.84, 0.0, 0.3))),
            transform: Transform::from_xyz(CHART_WIDTH / 2.0, CHART_HEIGHT / 2.0, 1.0),
            ..default()
        });

        // "Real" particles
        for _ in 0..30 {
            let x = CHART_WIDTH / 2.0 + (rng.gen::<f32>() - 0.5) * 150.0;
            let y = CHART_HEIGHT / 2.0 + (rng.gen::<f32>() - 0.5) * 150.0;
            let radius = rng.gen::<f32>() * 3.0 + 1.0;

            parent.spawn(MaterialMesh2dBundle {
                mesh: meshes.add(shape::Circle::new(radius).into()).into(),
                material: materials.add(ColorMaterial::from(Color::WHITE)),
                transform: Transform::from_xyz(x, y, 2.0),
                ..default()
            });
        }

        // Title
        parent.spawn(TextBundle::from_section(
            "Quantum Foam + Consciousness",
            TextStyle {
                font_size: 16.0,
                color: Color::WHITE,
                ..default()
            },
        ).with_text_alignment(TextAlignment::Center)
          .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(-30.0),
            width: Val::Percent(100.0),
            ..default()
        }));
    });

    // Panel 5: Correlation Chart
    commands.spawn((
        VisualizationPanel,
        NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                left: Val::Px(500.0),
                top: Val::Px(400.0),
                width: Val::Px(CHART_WIDTH),
                height: Val::Px(CHART_HEIGHT),
                ..default()
            },
            background_color: BackgroundColor(Color::rgba(1.0, 0.98, 0.88, 1.0)),
            ..default()
        },
    )).with_children(|parent| {
        // Title
        parent.spawn(TextBundle::from_section(
            "Consciousness vs Manifestation",
            TextStyle {
                font_size: 16.0,
                color: Color::BLACK,
                ..default()
            },
        ).with_text_alignment(TextAlignment::Center)
          .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(-30.0),
            width: Val::Percent(100.0),
            ..default()
        }));
    });

    // Panel 6: Summary Box
    commands.spawn((
        VisualizationPanel,
        NodeBundle {
            style: Style {
                position_type: PositionType::Absolute,
                left: Val::Px(950.0),
                top: Val::Px(400.0),
                width: Val::Px(CHART_WIDTH),
                height: Val::Px(CHART_HEIGHT),
                padding: UiRect::all(Val::Px(20.0)),
                ..default()
            },
            background_color: BackgroundColor(Color::rgba(1.0, 0.98, 0.88, 1.0)),
            ..default()
        },
    )).with_children(|parent| {
        // Title
        parent.spawn(TextBundle::from_section(
            "QUANTUM FOAM RESULTS",
            TextStyle {
                font_size: 14.0,
                color: Color::rgb(0.55, 0.27, 0.07),
                ..default()
            },
        ));

        // Summary text
        let total_particles: f32 = sim_data.timeline.iter().map(|d| d.particles).sum();
        let peak_rate = sim_data.timeline.iter().map(|d| d.particles).fold(0.0, f32::max);
        let avg_rate = total_particles / sim_data.timeline.len() as f32;

        let summary_text = format!(
            "Statistics:\n\
            Total particles: {:.0}\n\
            Peak rate: {:.1}/sec\n\
            Average rate: {:.1}/sec\n\n\
            Key Insight:\n\
            Attention creates reality.\n\
            Consciousness stabilizes\n\
            quantum fluctuations.",
            total_particles, peak_rate, avg_rate
        );

        parent.spawn(TextBundle::from_section(
            summary_text,
            TextStyle {
                font_size: 12.0,
                color: Color::BLACK,
                ..default()
            },
        ).with_style(Style {
            margin: UiRect::top(Val::Px(20.0)),
            ..default()
        }));
    });
}
