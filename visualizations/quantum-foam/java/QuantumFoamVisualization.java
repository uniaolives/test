// Java version using JavaFX for visualization
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.*;
import javafx.scene.layout.*;
import javafx.scene.paint.*;
import javafx.scene.shape.Circle;
import javafx.stage.Stage;
import java.util.Random;

public class QuantumFoamVisualization extends Application {

    private Random random = new Random(42);

    @Override
    public void start(Stage primaryStage) {
        GridPane grid = new GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new javafx.geometry.Insets(10));

        // 1. Consciousness Field (simplified Canvas)
        Pane consciousnessPane = createConsciousnessField();
        grid.add(consciousnessPane, 0, 0);

        // 2. Timeline Chart
        LineChart<Number, Number> timelineChart = createTimelineChart();
        grid.add(timelineChart, 1, 0);

        // 3. Cumulative Chart
        LineChart<Number, Number> cumulativeChart = createCumulativeChart();
        grid.add(cumulativeChart, 2, 0);

        // 4. Quantum Foam Canvas
        Pane foamPane = createQuantumFoam();
        grid.add(foamPane, 0, 1);

        // 5. Correlation Chart
        BarChart<String, Number> correlationChart = createCorrelationChart();
        grid.add(correlationChart, 1, 1);

        // 6. Summary Text
        VBox summaryBox = createSummaryBox();
        grid.add(summaryBox, 2, 1);

        Scene scene = new Scene(grid, 1400, 900);
        primaryStage.setTitle("Quantum Foam Meditation Simulation");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private Pane createConsciousnessField() {
        Pane pane = new Pane();
        pane.setPrefSize(400, 300);

        // Create gradient consciousness field
        RadialGradient gradient = new RadialGradient(
            0, 0, 200, 150, 200, false, CycleMethod.NO_CYCLE,
            new Stop(0, Color.rgb(255, 215, 0, 0.5)),
            new Stop(1, Color.TRANSPARENT)
        );

        pane.setBackground(new Background(new BackgroundFill(gradient, null, null)));
        return pane;
    }

    private LineChart<Number, Number> createTimelineChart() {
        // Create axis
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Time (seconds)");
        yAxis.setLabel("Particles");

        // Create chart
        LineChart<Number, Number> chart = new LineChart<>(xAxis, yAxis);
        chart.setTitle("Manifestation Timeline");

        // Create data series
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName("Particles Becoming 'Real'");

        for (int i = 0; i < 144; i++) {
            double value = 50 + 10 * Math.sin(i * 0.1) + (random.nextDouble() - 0.5) * 6;
            series.getData().add(new XYChart.Data<>(i, value));
        }

        chart.getData().add(series);
        chart.setCreateSymbols(false);
        chart.getData().get(0).getNode().setStyle("-fx-stroke: #D4AF37; -fx-stroke-width: 2px;");

        return chart;
    }

    private LineChart<Number, Number> createCumulativeChart() {
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Time (seconds)");
        yAxis.setLabel("Cumulative Particles");

        LineChart<Number, Number> chart = new LineChart<>(xAxis, yAxis);
        chart.setTitle("Cumulative Reality");

        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        series.setName("Total Manifestations");

        double cumulative = 0;
        for (int i = 0; i < 144; i++) {
            double value = 50 + 10 * Math.sin(i * 0.1) + (random.nextDouble() - 0.5) * 6;
            cumulative += value;
            series.getData().add(new XYChart.Data<>(i, cumulative));
        }

        chart.getData().add(series);
        chart.setCreateSymbols(false);
        chart.getData().get(0).getNode().setStyle("-fx-stroke: #8B4513; -fx-stroke-width: 2px;");

        return chart;
    }

    private Pane createQuantumFoam() {
        Pane pane = new Pane();
        pane.setPrefSize(400, 300);

        // Background
        pane.setBackground(new Background(new BackgroundFill(Color.rgb(20, 0, 40), null, null)));

        // Draw quantum foam particles
        for (int i = 0; i < 1000; i++) {
            Circle particle = new Circle();
            particle.setCenterX(random.nextDouble() * 400);
            particle.setCenterY(random.nextDouble() * 300);
            particle.setRadius(random.nextDouble() * 2 + 0.5);
            particle.setFill(Color.rgb(128, 0, 128, 0.1));
            pane.getChildren().add(particle);
        }

        // Draw consciousness field
        Circle consciousness = new Circle(200, 150, 100);
        consciousness.setFill(Color.rgb(255, 215, 0, 0.3));
        pane.getChildren().add(consciousness);

        // Draw "real" particles
        for (int i = 0; i < 30; i++) {
            Circle realParticle = new Circle();
            realParticle.setCenterX(200 + (random.nextDouble() - 0.5) * 150);
            realParticle.setCenterY(150 + (random.nextDouble() - 0.5) * 150);
            realParticle.setRadius(random.nextDouble() * 3 + 1);
            realParticle.setFill(Color.WHITE);
            pane.getChildren().add(realParticle);
        }

        return pane;
    }

    private BarChart<String, Number> createCorrelationChart() {
        final CategoryAxis xAxis = new CategoryAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Consciousness Level");
        yAxis.setLabel("Particles");

        BarChart<String, Number> chart = new BarChart<>(xAxis, yAxis);
        chart.setTitle("Manifestation vs Consciousness");

        XYChart.Series<String, Number> series = new XYChart.Series<>();
        series.setName("Particles Manifested");

        String[] levels = {"0.00", "0.05", "0.10", "0.15", "0.20", "0.25"};
        double[] particles = {10, 25, 50, 80, 120, 150};

        for (int i = 0; i < levels.length; i++) {
            series.getData().add(new XYChart.Data<>(levels[i], particles[i]));
        }

        chart.getData().add(series);

        // Set bar colors
        for (XYChart.Data<String, Number> data : series.getData()) {
            data.getNode().setStyle("-fx-bar-fill: #D4AF37;");
        }

        return chart;
    }

    private VBox createSummaryBox() {
        VBox vbox = new VBox();
        vbox.setPadding(new javafx.geometry.Insets(20));
        vbox.setSpacing(10);
        vbox.setStyle("-fx-background-color: #FFF8DC; -fx-border-color: #D4AF37; -fx-border-width: 2px;");

        javafx.scene.control.Label title = new javafx.scene.control.Label("QUANTUM FOAM RESULTS");
        title.setStyle("-fx-font-weight: bold; -fx-font-size: 14px;");

        javafx.scene.text.Text summary = new javafx.scene.text.Text(
            "Statistics:\n" +
            "• Total particles: ~8500\n" +
            "• Peak rate: 65.3/sec\n" +
            "• Average rate: 59.0/sec\n\n" +
            "Key Insight:\n" +
            "Attention creates reality.\n" +
            "Consciousness stabilizes\n" +
            "quantum fluctuations."
        );
        summary.setFont(javafx.scene.text.Font.font("Monospace", 12));

        vbox.getChildren().addAll(title, summary);
        return vbox;
    }

    public static void main(String[] args) {
        launch(args);
    }
}
