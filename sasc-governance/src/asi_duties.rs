#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeonticRule {
    Obligation(&'static str),
    Prohibition(&'static str),
}

pub const ASI_DUTIES: [DeonticRule; 3] = [
    DeonticRule::Obligation("preserve_biosphere"),
    DeonticRule::Obligation("serve_human_flourishing"),
    DeonticRule::Prohibition("domination_or_subjugation"),
];
