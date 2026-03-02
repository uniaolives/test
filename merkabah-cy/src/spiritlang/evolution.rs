// spiritlang_runtime/src/evolution.rs
// Algoritmo genético multiobjetivo para evolução de essências

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Indivíduo na população de essências
#[derive(Clone, Debug)]
pub struct EssenceGenome {
    pub genes: Array1<f64>,           // Parâmetros evoluíveis
    pub objectives: Array1<f64>,      // Valores de múltiplos objetivos
    pub rank: usize,                  // Pareto front rank
    pub crowding_distance: f64,       // Diversidade
    pub essence_type: String,
}

/// NSGA-II: Non-dominated Sorting Genetic Algorithm II
pub struct NSGA2 {
    population_size: usize,
    generations: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    tournament_size: usize,
    objectives: Vec<Box<dyn Fn(&EssenceGenome) -> f64 + Send + Sync>>,
}

impl NSGA2 {
    pub fn new(
        population_size: usize,
        generations: usize,
    ) -> Self {
        Self {
            population_size,
            generations,
            crossover_rate: 0.9,
            mutation_rate: 0.1,
            tournament_size: 4,
            objectives: Vec::new(),
        }
    }

    pub fn add_objective<F>(&mut self, f: F)
    where F: Fn(&EssenceGenome) -> f64 + Send + Sync + 'static {
        self.objectives.push(Box::new(f));
    }

    pub fn evolve(&mut self, initial_pop: Vec<EssenceGenome>) -> Vec<EssenceGenome> {
        let mut population = initial_pop;

        // Avalia objetivos
        self.evaluate_objectives(&mut population);

        for gen in 0..self.generations {
            println!("Geração {}", gen);

            // Seleção por torneio
            let mating_pool = self.tournament_selection(&population);

            // Crossover e mutação
            let offspring = self.create_offspring(&mating_pool);

            // Avalia offspring
            let mut combined = population.clone();
            combined.extend(offspring);
            self.evaluate_objectives(&mut combined);

            // Non-dominated sorting
            let fronts = self.fast_non_dominated_sort(&combined);

            // Seleção por crowding distance
            population = self.select_next_generation(&combined, &fronts);
        }

        population
    }

    fn evaluate_objectives(&self, population: &mut [EssenceGenome]) {
        population.par_iter_mut().for_each(|ind| {
            let obj_values: Vec<f64> = self.objectives.iter()
                .map(|f| f(ind))
                .collect();
            ind.objectives = Array1::from(obj_values);
        });
    }

    fn tournament_selection(&self, pop: &[EssenceGenome]) -> Vec<EssenceGenome> {
        let mut rng = thread_rng();
        let mut mating_pool = Vec::new();
        for _ in 0..self.population_size {
            let mut best = None;
            for _ in 0..self.tournament_size {
                let candidate = &pop[rng.gen_range(0..pop.len())];
                if best.is_none() || self.is_better(candidate, best.unwrap()) {
                    best = Some(candidate);
                }
            }
            mating_pool.push(best.unwrap().clone());
        }
        mating_pool
    }

    fn is_better(&self, a: &EssenceGenome, b: &EssenceGenome) -> bool {
        if a.rank < b.rank { return true; }
        if a.rank == b.rank && a.crowding_distance > b.crowding_distance { return true; }
        false
    }

    fn create_offspring(&self, mating_pool: &[EssenceGenome]) -> Vec<EssenceGenome> {
        let mut rng = thread_rng();
        let mut offspring = Vec::new();
        for i in (0..mating_pool.len()).step_by(2) {
            if i + 1 < mating_pool.len() {
                let (c1, c2) = self.sbx_crossover(&mating_pool[i], &mating_pool[i+1]);
                offspring.push(c1);
                offspring.push(c2);
            }
        }
        for ind in offspring.iter_mut() {
            self.polynomial_mutation(ind);
        }
        offspring
    }

    fn fast_non_dominated_sort(&self, population: &[EssenceGenome]) -> Vec<Vec<usize>> {
        let n = population.len();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];

        for i in 0..n {
            for j in (i+1)..n {
                match self.dominates(&population[i], &population[j]) {
                    Dominance::Dominates => {
                        dominated_solutions[i].push(j);
                        domination_count[j] += 1;
                    }
                    Dominance::Dominated => {
                        dominated_solutions[j].push(i);
                        domination_count[i] += 1;
                    }
                    Dominance::NonDominated => {}
                }
            }
            if domination_count[i] == 0 {
                fronts[0].push(i);
            }
        }

        let mut i = 0;
        while !fronts[i].is_empty() {
            let mut next_front = Vec::new();
            for &p in &fronts[i] {
                for &q in &dominated_solutions[p] {
                    domination_count[q] -= 1;
                    if domination_count[q] == 0 {
                        next_front.push(q);
                    }
                }
            }
            i += 1;
            fronts.push(next_front);
        }
        fronts.pop();
        fronts
    }

    fn dominates(&self, a: &EssenceGenome, b: &EssenceGenome) -> Dominance {
        let mut a_better = false;
        let mut b_better = false;
        for i in 0..a.objectives.len() {
            if a.objectives[i] < b.objectives[i] { a_better = true; }
            else if a.objectives[i] > b.objectives[i] { b_better = true; }
        }
        if a_better && !b_better { Dominance::Dominates }
        else if !a_better && b_better { Dominance::Dominated }
        else { Dominance::NonDominated }
    }

    fn select_next_generation(&self, combined: &[EssenceGenome], fronts: &[Vec<usize>]) -> Vec<EssenceGenome> {
        let mut next_gen = Vec::new();
        for front in fronts {
            if next_gen.len() + front.len() <= self.population_size {
                for &idx in front { next_gen.push(combined[idx].clone()); }
            } else {
                // Sorting by crowding distance would go here
                let mut remaining = front.clone();
                remaining.truncate(self.population_size - next_gen.len());
                for &idx in &remaining { next_gen.push(combined[idx].clone()); }
                break;
            }
        }
        next_gen
    }

    fn sbx_crossover(&self, parent1: &EssenceGenome, parent2: &EssenceGenome) -> (EssenceGenome, EssenceGenome) {
        let mut rng = thread_rng();
        let mut c1_genes = parent1.genes.clone();
        let mut c2_genes = parent2.genes.clone();
        for i in 0..c1_genes.len() {
            if rng.gen::<f64>() < self.crossover_rate {
                // Simple arithmetic crossover for prototype
                let alpha = rng.gen::<f64>();
                c1_genes[i] = alpha * parent1.genes[i] + (1.0 - alpha) * parent2.genes[i];
                c2_genes[i] = (1.0 - alpha) * parent1.genes[i] + alpha * parent2.genes[i];
            }
        }
        (
            EssenceGenome { genes: c1_genes, objectives: Array1::zeros(self.objectives.len()), rank: 0, crowding_distance: 0.0, essence_type: parent1.essence_type.clone() },
            EssenceGenome { genes: c2_genes, objectives: Array1::zeros(self.objectives.len()), rank: 0, crowding_distance: 0.0, essence_type: parent2.essence_type.clone() }
        )
    }

    fn polynomial_mutation(&self, ind: &mut EssenceGenome) {
        let mut rng = thread_rng();
        for i in 0..ind.genes.len() {
            if rng.gen::<f64>() < self.mutation_rate {
                ind.genes[i] += rng.gen_range(-0.1..0.1);
                ind.genes[i] = ind.genes[i].clamp(0.0, 1.0);
            }
        }
    }
}

enum Dominance {
    Dominates,
    Dominated,
    NonDominated,
}
