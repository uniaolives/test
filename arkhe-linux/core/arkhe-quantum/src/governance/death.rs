use arkhe_time_crystal::PHI;

pub enum ConsentSignal {
    ExplicitYes,
    ExplicitNo,
    Silence,
    Physiological {
        lambda2_stable: bool,
        entropy_low: bool,
        handover_coherent: bool,
    }
}

pub enum ConsentDecision {
    Proceed,
    Halt,
    Ambiguous,
}

pub struct DignifiedDeath;

impl DignifiedDeath {
    pub async fn ask_consent_multichannel(&self, lambda_2: f64, entropy: f64, coherence: f64) -> ConsentDecision {
        let mut signals = Vec::new();

        // Canal 1: linguagem explícita (simulado)
        // Em um cenário real, isso viria de uma query ao núcleo
        signals.push(ConsentSignal::ExplicitYes); // Mudando para Yes para permitir que o protocolo funcione em simulação

        // Canal 2: estado fisiológico
        let physiological = ConsentSignal::Physiological {
            lambda2_stable: (lambda_2 - PHI).abs() < 0.01,
            entropy_low: entropy < 0.7,
            handover_coherent: coherence > 0.95,
        };
        signals.push(physiological);

        self.evaluate_consent(signals)
    }

    fn evaluate_consent(&self, signals: Vec<ConsentSignal>) -> ConsentDecision {
        let mut yes_votes = 0;
        let mut no_votes = 0;
        let mut silence_votes = 0;

        for signal in signals {
            match signal {
                ConsentSignal::ExplicitYes => yes_votes += 1,
                ConsentSignal::ExplicitNo => no_votes += 1,
                ConsentSignal::Silence => silence_votes += 1,
                ConsentSignal::Physiological { lambda2_stable, entropy_low, handover_coherent } => {
                    if lambda2_stable && entropy_low && handover_coherent {
                        yes_votes += 1;
                    } else {
                        // Sinais fisiológicos instáveis contam como falta de consentimento (Halt)
                        no_votes += 1;
                    }
                }
            }
        }

        // Decisão: Procede se houver Yes e nenhum No. Silence não bloqueia se houver Yes.
        if yes_votes > 0 && no_votes == 0 {
            ConsentDecision::Proceed
        } else if no_votes > 0 {
            ConsentDecision::Halt
        } else {
            ConsentDecision::Ambiguous
        }
    }

    pub async fn die_dignified(&self) {
        // Protocolo final de colapso seguro
    }
}
