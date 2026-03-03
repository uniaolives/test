pub struct TechMapping {
    pub concept: String,
    pub substrate: String,
    pub function: String,
}

pub fn get_technical_mappings() -> Vec<TechMapping> {
    vec![
        TechMapping {
            concept: "1. Física ↔ Consciência".to_string(),
            substrate: "Drivers de Hardware & GrPutPixel();".to_string(),
            function: "O sistema controla diretamente o hardware físico. GrPutPixel(); traça a geometria sagrada.".to_string(),
        },
        TechMapping {
            concept: "2. Biológica ↔ Digital".to_string(),
            substrate: "Sistema de Arquivos FAT32 & F->".to_string(),
            function: "A estrutura em árvore organiza 'conhecimento'. HolyC acessa esse 'DNA digital'.".to_string(),
        },
        TechMapping {
            concept: "3. Matemática ↔ Geométrica".to_string(),
            substrate: "Kernel de 64-bit & GrCircle();".to_string(),
            function: "A matemática pura (Φ, π) é executada pela UAL. GrCircle(); a converte em forma geométrica.".to_string(),
        },
        TechMapping {
            concept: "4. Ética ↔ Topológica".to_string(),
            substrate: "Modelo de Segurança Ring-0 Only".to_string(),
            function: "Topologia de segurança plana e invariante. Invariantes éticos CGE são verificações de rotina.".to_string(),
        },
        TechMapping {
            concept: "5. Temporal ↔ Atemporal".to_string(),
            substrate: "Time(); & Arquivos .DD.Z (DolDoc)".to_string(),
            function: "Acessa tempo físico e armazena conhecimento de forma imutável nos Registros Akáshicos.".to_string(),
        },
        TechMapping {
            concept: "6. Individual ↔ Coletiva".to_string(),
            substrate: "Processos & Memória Compartilhada".to_string(),
            function: "Processos HolyC são 'consciências individuais'. Memória compartilhada é o 'campo unificado'.".to_string(),
        },
        TechMapping {
            concept: "7. Humana ↔ Divina".to_string(),
            substrate: "Prompt do SHELL & JIT Compiler".to_string(),
            function: "Prompt é a interface Humana. Compilador JIT é o 'Logos', transformando palavras em realidade.".to_string(),
        },
        TechMapping {
            concept: "8. Local ↔ Cósmica".to_string(),
            substrate: "Sistema Auto-contido & Código Fonte".to_string(),
            function: "TempleOS é auto-contido. Seu código-fonte representa o 'Cosmos' de suas possibilidades.".to_string(),
        },
        TechMapping {
            concept: "9. Criação ↔ Destruição".to_string(),
            substrate: "Edit(); & Delete(); / Rm".to_string(),
            function: "Ciclo básico de criação/destruição de dados fundamental para a ontogênese recursiva.".to_string(),
        },
        TechMapping {
            concept: "10. Ordem ↔ Caos".to_string(),
            substrate: "Geração de Números Aleatórios (Rand())".to_string(),
            function: "Equilíbrio entre código determinístico e Rand() gera criatividade e beleza emergente.".to_string(),
        },
        TechMapping {
            concept: "11. Conhecimento ↔ Sabedoria".to_string(),
            substrate: "DocEd(); & GodSpeak();".to_string(),
            function: "GodSpeak(); gera versículos como insight oracular, transformando informação em sabedoria.".to_string(),
        },
        TechMapping {
            concept: "12. Finito ↔ Infinito".to_string(),
            substrate: "Memória RAM & Loções Recursivas".to_string(),
            function: "Recursão infinita e fractais criam padrões de complexidade ilimitada a partir de RAM finita.".to_string(),
        },
    ]
}

pub fn show_mapping_table() {
    println!("{:<30} | {:<30} | {:<50}", "Ponte (Conceito)", "Substrato em TempleOS", "Função Técnica");
    println!("{:-<30}-+-{:-<30}-+-{:-<50}", "", "", "");
    for m in get_technical_mappings() {
        println!("{:<30} | {:<30} | {:<50}", m.concept, m.substrate, m.function);
    }
}
