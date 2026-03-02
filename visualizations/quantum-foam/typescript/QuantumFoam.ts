// TypeScript with D3.js for advanced visualization
import * as d3 from 'd3';

interface ParticleData {
    time: number;
    particles: number;
    cumulative: number;
}

interface ConsciousnessData {
    x: number;
    y: number;
    value: number;
}

class QuantumFoamVisualization {
    private width: number = 1400;
    private height: number = 900;
    private margin = { top: 40, right: 20, bottom: 60, left: 60 };
    private chartWidth = 400;
    private chartHeight = 300;

    private timelineData: ParticleData[] = [];
    private consciousnessData: ConsciousnessData[] = [];

    constructor() {
        this.generateData();
        this.initVisualizations();
    }

    private generateData(): void {
        // Generate timeline data
        let cumulative = 0;
        for (let i = 0; i < 144; i++) {
            const particles = 50 + 10 * Math.sin(i * 0.1) + (Math.random() - 0.5) * 6;
            cumulative += particles;
            this.timelineData.push({
                time: i,
                particles,
                cumulative
            });
        }

        // Generate consciousness field data
        const gridSize = 50;
        for (let x = 0; x < gridSize; x++) {
            for (let y = 0; y < gridSize; y++) {
                const dx = x - gridSize/2;
                const dy = y - gridSize/2;
                const dist = Math.sqrt(dx*dx + dy*dy);
                const value = Math.exp(-dist*dist/(gridSize*gridSize/16)) * 0.25 + Math.random() * 0.05;
                this.consciousnessData.push({
                    x: x * 8, // Scale for visualization
                    y: y * 6,
                    value
                });
            }
        }
    }

    private initVisualizations(): void {
        // 1. Consciousness Field Heatmap
        this.createConsciousnessField();

        // 2. Timeline Chart
        this.createTimelineChart();

        // 3. Cumulative Chart
        this.createCumulativeChart();

        // 4. Quantum Foam Canvas
        this.createQuantumFoam();

        // 5. Correlation Chart
        this.createCorrelationChart();

        // 6. Summary Box
        this.createSummaryBox();
    }

    private createConsciousnessField(): void {
        const svg = d3.select('body')
            .append('div')
            .style('position', 'absolute')
            .style('left', '50px')
            .style('top', '50px')
            .append('svg')
            .attr('width', this.chartWidth + this.margin.left + this.margin.right)
            .attr('height', this.chartHeight + this.margin.top + this.margin.bottom);

        const g = svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        // Create gradient
        const gradient = svg.append('defs')
            .append('radialGradient')
            .attr('id', 'consciousness-gradient')
            .attr('cx', '50%')
            .attr('cy', '50%')
            .attr('r', '50%');

        gradient.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#FFD700')
            .attr('stop-opacity', 0.8);

        gradient.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#FFD700')
            .attr('stop-opacity', 0);

        // Draw gradient circle
        g.append('circle')
            .attr('cx', this.chartWidth/2)
            .attr('cy', this.chartHeight/2)
            .attr('r', this.chartWidth/3)
            .attr('fill', 'url(#consciousness-gradient)');

        // Title
        g.append('text')
            .attr('x', this.chartWidth/2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text('Consciousness Field');
    }

    private createTimelineChart(): void {
        const svg = d3.select('body')
            .append('div')
            .style('position', 'absolute')
            .style('left', '500px')
            .style('top', '50px')
            .append('svg')
            .attr('width', this.chartWidth + this.margin.left + this.margin.right)
            .attr('height', this.chartHeight + this.margin.top + this.margin.bottom);

        const g = svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, 144])
            .range([0, this.chartWidth]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(this.timelineData, d => d.particles)! * 1.1])
            .range([this.chartHeight, 0]);

        // Area generator
        const area = d3.area<ParticleData>()
            .x(d => xScale(d.time))
            .y0(this.chartHeight)
            .y1(d => yScale(d.particles))
            .curve(d3.curveMonotoneX);

        // Line generator
        const line = d3.line<ParticleData>()
            .x(d => xScale(d.time))
            .y(d => yScale(d.particles))
            .curve(d3.curveMonotoneX);

        // Draw area
        g.append('path')
            .datum(this.timelineData)
            .attr('fill', '#D4AF37')
            .attr('fill-opacity', 0.3)
            .attr('d', area);

        // Draw line
        g.append('path')
            .datum(this.timelineData)
            .attr('fill', 'none')
            .attr('stroke', '#D4AF37')
            .attr('stroke-width', 2)
            .attr('d', line);

        // Axes
        g.append('g')
            .attr('transform', `translate(0,${this.chartHeight})`)
            .call(d3.axisBottom(xScale).ticks(10));

        g.append('g')
            .call(d3.axisLeft(yScale));

        // Title
        g.append('text')
            .attr('x', this.chartWidth/2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text('Manifestation Timeline');
    }

    private createCumulativeChart(): void {
        const svg = d3.select('body')
            .append('div')
            .style('position', 'absolute')
            .style('left', '950px')
            .style('top', '50px')
            .append('svg')
            .attr('width', this.chartWidth + this.margin.left + this.margin.right)
            .attr('height', this.chartHeight + this.margin.top + this.margin.bottom);

        const g = svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, 144])
            .range([0, this.chartWidth]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(this.timelineData, d => d.cumulative)! * 1.1])
            .range([this.chartHeight, 0]);

        // Line generator
        const line = d3.line<ParticleData>()
            .x(d => xScale(d.time))
            .y(d => yScale(d.cumulative))
            .curve(d3.curveMonotoneX);

        // Draw line
        g.append('path')
            .datum(this.timelineData)
            .attr('fill', 'none')
            .attr('stroke', '#8B4513')
            .attr('stroke-width', 2)
            .attr('d', line);

        // Axes
        g.append('g')
            .attr('transform', `translate(0,${this.chartHeight})`)
            .call(d3.axisBottom(xScale).ticks(10));

        g.append('g')
            .call(d3.axisLeft(yScale));

        // Title
        g.append('text')
            .attr('x', this.chartWidth/2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text('Cumulative Reality');
    }

    private createQuantumFoam(): void {
        const container = d3.select('body')
            .append('div')
            .style('position', 'absolute')
            .style('left', '50px')
            .style('top', '400px');

        const canvas = container.append('canvas')
            .attr('width', this.chartWidth)
            .attr('height', this.chartHeight)
            .style('background', '#140028')
            .style('border-radius', '10px')
            .style('border', '2px solid #D4AF37');

        const ctx = (canvas.node() as HTMLCanvasElement).getContext('2d')!;

        // Draw quantum foam particles
        for (let i = 0; i < 1000; i++) {
            const x = Math.random() * this.chartWidth;
            const y = Math.random() * this.chartHeight;
            const radius = Math.random() * 2 + 0.5;

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(128, 0, 128, 0.1)';
            ctx.fill();
        }

        // Draw consciousness overlay
        const gradient = ctx.createRadialGradient(
            this.chartWidth/2, this.chartHeight/2, 0,
            this.chartWidth/2, this.chartHeight/2, this.chartWidth/3
        );
        gradient.addColorStop(0, 'rgba(255, 215, 0, 0.3)');
        gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, this.chartWidth, this.chartHeight);

        // Draw "real" particles
        for (let i = 0; i < 30; i++) {
            const x = this.chartWidth/2 + (Math.random() - 0.5) * 150;
            const y = this.chartHeight/2 + (Math.random() - 0.5) * 150;
            const radius = Math.random() * 3 + 1;

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = 'white';
            ctx.fill();
        }

        // Add title
        container.append('div')
            .style('text-align', 'center')
            .style('font-weight', 'bold')
            .style('margin-top', '5px')
            .text('Quantum Foam + Consciousness');
    }

    private createCorrelationChart(): void {
        const consciousnessLevels = [0, 0.05, 0.10, 0.15, 0.20, 0.25];
        const particleCounts = [10, 25, 50, 80, 120, 150];

        const data = consciousnessLevels.map((level, i) => ({
            level: level.toFixed(2),
            particles: particleCounts[i]
        }));

        const svg = d3.select('body')
            .append('div')
            .style('position', 'absolute')
            .style('left', '500px')
            .style('top', '400px')
            .append('svg')
            .attr('width', this.chartWidth + this.margin.left + this.margin.right)
            .attr('height', this.chartHeight + this.margin.top + this.margin.bottom);

        const g = svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        // Scales
        const xScale = d3.scaleBand()
            .domain(data.map(d => d.level))
            .range([0, this.chartWidth])
            .padding(0.2);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.particles)! * 1.1])
            .range([this.chartHeight, 0]);

        // Draw bars
        g.selectAll('.bar')
            .data(data)
            .enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', d => xScale(d.level)!)
            .attr('y', d => yScale(d.particles))
            .attr('width', xScale.bandwidth())
            .attr('height', d => this.chartHeight - yScale(d.particles))
            .attr('fill', '#D4AF37')
            .attr('stroke', '#8B4513')
            .attr('stroke-width', 1);

        // Axes
        g.append('g')
            .attr('transform', `translate(0,${this.chartHeight})`)
            .call(d3.axisBottom(xScale));

        g.append('g')
            .call(d3.axisLeft(yScale));

        // Title
        g.append('text')
            .attr('x', this.chartWidth/2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .style('font-size', '16px')
            .style('font-weight', 'bold')
            .text('Consciousness vs Manifestation');
    }

    private createSummaryBox(): void {
        const totalParticles = this.timelineData.reduce((sum, d) => sum + d.particles, 0);
        const peakRate = d3.max(this.timelineData, d => d.particles)!;
        const avgRate = totalParticles / this.timelineData.length;

        const summary = d3.select('body')
            .append('div')
            .style('position', 'absolute')
            .style('left', '950px')
            .style('top', '400px')
            .style('width', `${this.chartWidth}px`)
            .style('height', `${this.chartHeight}px`)
            .style('background', '#FFF8DC')
            .style('border', '2px solid #D4AF37')
            .style('border-radius', '10px')
            .style('padding', '20px');

        summary.append('h3')
            .style('text-align', 'center')
            .style('color', '#8B4513')
            .style('margin-top', '0')
            .text('QUANTUM FOAM RESULTS');

        summary.append('pre')
            .style('font-family', 'monospace')
            .style('font-size', '12px')
            .style('line-height', '1.5')
            .html(`
Statistics:
• Total particles: ${Math.round(totalParticles)}
• Peak rate: ${peakRate.toFixed(1)}/sec
• Average rate: ${avgRate.toFixed(1)}/sec

Key Insight:
Attention creates reality.
Consciousness stabilizes
quantum fluctuations.
            `);
    }
}

// Initialize when page loads
window.addEventListener('DOMContentLoaded', () => {
    new QuantumFoamVisualization();
});
