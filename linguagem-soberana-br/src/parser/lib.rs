// src/parser/lib.rs
// Parser da Linguagem Soberana - Fase A (ICMS)
// Converte sintaxe em português para AST verificável

use logos::Logos;

/// Tokens léxicos da Linguagem Soberana
#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")] // Ignorar espaços
#[logos(skip r"//[^\n]*")]   // Ignorar comentários
pub enum Token {
    // Palavras-chave
    #[token("funcao")]
    Funcao,
    #[token("processo")]
    Processo,
    #[token("retorne")]
    Retorne,
    #[token("se")]
    Se,
    #[token("entao")]
    Entao,
    #[token("senao")]
    Senao,
    #[token("para")]
    Para,
    #[token("cada")]
    Cada,
    #[token("em")]
    Em,
    #[token("enquanto")]
    Enquanto,
    #[token("etapa")]
    Etapa,

    // Anotações constitucionais
    #[token("verificavel_mathematicamente")]
    VerificavelMathematicamente,
    #[token("com_precisao")]
    ComPrecisao,
    #[token("com_complexidade")]
    ComComplexidade,
    #[token("com_rastreamento_completo")]
    ComRastreamentoCompleto,
    #[token("auditavel_por")]
    AuditavelPor,
    #[token("sujeito_a_voto")]
    SujeitoAVoto,

    // Tipos
    #[token("Decimal")]
    Decimal,
    #[token("Inteiro")]
    Inteiro,
    #[token("Booleano")]
    Booleano,
    #[token("Texto")]
    Texto,
    #[token("Lista")]
    Lista,
    #[token("Mapeamento")]
    Mapeamento,

    // Operadores
    #[token("+")]
    Mais,
    #[token("-")]
    Menos,
    #[token("*")]
    Vezes,
    #[token("/")]
    Dividido,
    #[token("==")]
    Igual,
    #[token("!=")]
    Diferente,
    #[token("<")]
    Menor,
    #[token(">")]
    Maior,
    #[token("<=")]
    MenorIgual,
    #[token(">=")]
    MaiorIgual,

    // Delimitadores
    #[token("(")]
    ParenEsq,
    #[token(")")]
    ParenDir,
    #[token("{")]
    ChaveEsq,
    #[token("}")]
    ChaveDir,
    #[token("[")]
    ColcheteEsq,
    #[token("]")]
    ColcheteDir,
    #[token("->")]
    Seta,
    #[token(":")]
    DoisPontos,
    #[token(",")]
    Virgula,
    #[token(";")]
    PontoVirgula,

    // Identificadores e literais
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Identificador(String),

    #[regex(r#""([^"\\]|\\.)*""#, |lex| lex.slice()[1..lex.slice().len()-1].to_string())]
    TextoLit(String),

    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().unwrap())]
    DecimalLit(f64),

    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().unwrap())]
    InteiroLit(i64),
}

/// AST (Árvore de Sintaxe Abstrata) Constitucional
#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    Identificador(String),
    Binaria {
        esq: Box<Expr>,
        op: Operador,
        dir: Box<Expr>,
    },
    Chamada {
        funcao: String,
        args: Vec<Expr>,
    },
}

#[derive(Debug, Clone)]
pub enum Literal {
    Decimal(f64, u8), // valor, precisão
    Inteiro(i64),
    Booleano(bool),
    Texto(String),
}

#[derive(Debug, Clone)]
pub enum Operador {
    Soma, Subtracao, Multiplicacao, Divisao,
    Igual, Diferente, Menor, Maior, MenorIgual, MaiorIgual,
}

#[derive(Debug, Clone)]
pub struct Funcao {
    pub nome: String,
    pub parametros: Vec<(String, Tipo)>,
    pub retorno: Tipo,
    pub anotacoes: Vec<Anotacao>,
    pub corpo: Vec<Comando>,
}

#[derive(Debug, Clone)]
pub struct Processo {
    pub nome: String,
    pub parametros: Vec<(String, Tipo)>,
    pub retorno: Tipo,
    pub anotacoes: Vec<Anotacao>,
    pub etapas: Vec<Etapa>,
}

#[derive(Debug, Clone)]
pub struct Etapa {
    pub nome: String,
    pub corpo: Vec<Comando>,
}

#[derive(Debug, Clone)]
pub enum Tipo {
    Decimal(u8), // precisão
    Inteiro,
    Booleano,
    Texto,
    Lista(Box<Tipo>),
    Mapeamento(Box<Tipo>, Box<Tipo>),
    Estrutura(String),
}

#[derive(Debug, Clone)]
pub enum Anotacao {
    VerificavelMathematicamente,
    Precisao(u8),
    Complexidade(String), // "O(n)", "O(n log n)", etc.
    RastreamentoCompleto(String), // invariante
    AuditavelPor(String),
    SujeitoAVoto(String, f64), // autoridade, quorum
}

#[derive(Debug, Clone)]
pub enum Comando {
    Atribuicao { var: String, expr: Expr },
    Retorno(Expr),
    Se { cond: Expr, entao: Vec<Comando>, senao: Option<Vec<Comando>> },
    ParaCada { var: String, colecao: String, corpo: Vec<Comando> },
    Expressao(Expr),
}

#[derive(Debug, Clone)]
pub enum Declaracao {
    Funcao(Funcao),
    Processo(Processo),
}

/// Parser recursivo descendente
pub struct Parser {
    tokens: Vec<Token>,
    posicao: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, posicao: 0 }
    }

    fn atual(&self) -> Option<&Token> {
        self.tokens.get(self.posicao)
    }

    fn avancar(&mut self) -> Option<&Token> {
        let tok = self.tokens.get(self.posicao);
        self.posicao += 1;
        tok
    }

    fn avancar_clonado(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.posicao).cloned();
        self.posicao += 1;
        tok
    }

    fn espera(&mut self, esperado: Token) -> Result<(), String> {
        match self.avancar() {
            Some(tok) if std::mem::discriminant(tok) == std::mem::discriminant(&esperado) => Ok(()),
            Some(tok) => Err(format!("Esperado {:?}, encontrado {:?}", esperado, tok)),
            None => Err("Fim inesperado do arquivo".to_string()),
        }
    }

    pub fn parse_declaracao(&mut self) -> Result<Declaracao, String> {
        match self.atual() {
            Some(Token::Funcao) => Ok(Declaracao::Funcao(self.parse_funcao()?)),
            Some(Token::Processo) => Ok(Declaracao::Processo(self.parse_processo()?)),
            _ => Err("Esperado 'funcao' ou 'processo'".to_string()),
        }
    }

    /// Parse de função completa
    pub fn parse_funcao(&mut self) -> Result<Funcao, String> {
        self.espera(Token::Funcao)?;

        let nome = match self.avancar() {
            Some(Token::Identificador(n)) => n.clone(),
            _ => return Err("Esperado identificador após 'funcao'".to_string()),
        };

        self.espera(Token::ParenEsq)?;
        let parametros = self.parse_parametros()?;
        self.espera(Token::ParenDir)?;

        self.espera(Token::Seta)?;
        let retorno = self.parse_tipo()?;

        let anotacoes = self.parse_anotacoes()?;
        let corpo = self.parse_bloco()?;

        Ok(Funcao {
            nome,
            parametros,
            retorno,
            anotacoes,
            corpo,
        })
    }

    pub fn parse_processo(&mut self) -> Result<Processo, String> {
        self.espera(Token::Processo)?;
        let nome = match self.avancar() {
            Some(Token::Identificador(n)) => n.clone(),
            _ => return Err("Esperado identificador após 'processo'".to_string()),
        };

        self.espera(Token::ParenEsq)?;
        let parametros = self.parse_parametros()?;
        self.espera(Token::ParenDir)?;

        self.espera(Token::Seta)?;
        let retorno = self.parse_tipo()?;

        let anotacoes = self.parse_anotacoes()?;

        self.espera(Token::ChaveEsq)?;
        let mut etapas = Vec::new();
        while let Some(tok) = self.atual() {
            if let Token::ChaveDir = tok {
                break;
            }
            etapas.push(self.parse_etapa()?);
        }
        self.espera(Token::ChaveDir)?;

        Ok(Processo {
            nome,
            parametros,
            retorno,
            anotacoes,
            etapas,
        })
    }

    fn parse_etapa(&mut self) -> Result<Etapa, String> {
        self.espera(Token::Etapa)?;
        let nome = match self.avancar() {
            Some(Token::Identificador(n)) => n.clone(),
            _ => return Err("Esperado identificador de etapa".to_string()),
        };
        let corpo = self.parse_bloco()?;
        Ok(Etapa { nome, corpo })
    }

    fn parse_parametros(&mut self) -> Result<Vec<(String, Tipo)>, String> {
        let mut params = Vec::new();

        while let Some(tok) = self.atual() {
            if let Token::Identificador(nome) = tok.clone() {
                self.avancar();
                self.espera(Token::DoisPontos)?;
                let tipo = self.parse_tipo()?;
                params.push((nome, tipo));

                if let Some(Token::Virgula) = self.atual() {
                    self.avancar();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(params)
    }

    fn parse_tipo(&mut self) -> Result<Tipo, String> {
        let tok = self.avancar_clonado().ok_or("Tipo esperado")?;
        match tok {
            Token::Decimal => {
                if let Some(Token::Menor) = self.atual() {
                    self.avancar();
                    if let Some(Token::InteiroLit(n)) = self.avancar() {
                        let val = *n;
                        self.espera(Token::Maior)?;
                        Ok(Tipo::Decimal(val as u8))
                    } else {
                        Err("Esperado número para precisão do Decimal".to_string())
                    }
                } else {
                    Ok(Tipo::Decimal(2))
                }
            }
            Token::Inteiro => Ok(Tipo::Inteiro),
            Token::Booleano => Ok(Tipo::Booleano),
            Token::Texto => Ok(Tipo::Texto),
            Token::Lista => {
                self.espera(Token::Menor)?;
                let inner = self.parse_tipo()?;
                self.espera(Token::Maior)?;
                Ok(Tipo::Lista(Box::new(inner)))
            }
            Token::Mapeamento => {
                self.espera(Token::Menor)?;
                let k = self.parse_tipo()?;
                self.espera(Token::Virgula)?;
                let v = self.parse_tipo()?;
                self.espera(Token::Maior)?;
                Ok(Tipo::Mapeamento(Box::new(k), Box::new(v)))
            }
            Token::Identificador(s) => Ok(Tipo::Estrutura(s)),
            _ => Err(format!("Tipo esperado, encontrado {:?}", tok)),
        }
    }

    fn parse_anotacoes(&mut self) -> Result<Vec<Anotacao>, String> {
        let mut anotacoes = Vec::new();

        while let Some(tok) = self.atual() {
            match tok {
                Token::VerificavelMathematicamente => {
                    self.avancar();
                    anotacoes.push(Anotacao::VerificavelMathematicamente);
                }
                Token::ComPrecisao => {
                    self.avancar();
                    self.espera(Token::ParenEsq)?;
                    match self.avancar() {
                        Some(Token::Identificador(s)) if s == "decimais" => {},
                        _ => return Err("Esperado 'decimais'".to_string()),
                    }
                    self.espera(Token::DoisPontos)?;
                    if let Some(Token::InteiroLit(n)) = self.avancar() {
                        anotacoes.push(Anotacao::Precisao(*n as u8));
                        self.espera(Token::ParenDir)?;
                    } else {
                        return Err("Esperado valor inteiro".to_string());
                    }
                }
                Token::AuditavelPor => {
                    self.avancar();
                    self.espera(Token::ParenEsq)?;
                    if let Some(Token::TextoLit(s)) = self.avancar() {
                        anotacoes.push(Anotacao::AuditavelPor(s.clone()));
                        self.espera(Token::ParenDir)?;
                    } else {
                        return Err("Esperado texto em auditavel_por".to_string());
                    }
                }
                _ => break,
            }
        }

        Ok(anotacoes)
    }

    fn parse_bloco(&mut self) -> Result<Vec<Comando>, String> {
        self.espera(Token::ChaveEsq)?;
        let mut comandos = Vec::new();

        while let Some(tok) = self.atual() {
            if let Token::ChaveDir = tok {
                break;
            }
            comandos.push(self.parse_comando()?);
        }

        self.espera(Token::ChaveDir)?;
        Ok(comandos)
    }

    fn parse_comando(&mut self) -> Result<Comando, String> {
        match self.atual() {
            Some(Token::Retorne) => {
                self.avancar();
                let expr = self.parse_expressao()?;
                if let Some(Token::PontoVirgula) = self.atual() {
                    self.avancar();
                }
                Ok(Comando::Retorno(expr))
            }
            Some(Token::Se) => self.parse_se(),
            Some(Token::Para) => self.parse_para_cada(),
            _ => {
                let expr = self.parse_expressao()?;
                if let Some(Token::PontoVirgula) = self.atual() {
                    self.avancar();
                }
                Ok(Comando::Expressao(expr))
            }
        }
    }

    fn parse_se(&mut self) -> Result<Comando, String> {
        self.espera(Token::Se)?;
        let cond = self.parse_expressao()?;
        self.espera(Token::Entao)?;
        let entao = self.parse_bloco()?;

        let senao = if let Some(Token::Senao) = self.atual() {
            self.avancar();
            Some(self.parse_bloco()?)
        } else {
            None
        };

        Ok(Comando::Se { cond, entao, senao })
    }

    fn parse_para_cada(&mut self) -> Result<Comando, String> {
        self.espera(Token::Para)?;
        self.espera(Token::Cada)?;

        let var = match self.avancar() {
            Some(Token::Identificador(v)) => v.clone(),
            _ => return Err("Esperado variável em 'para cada'".to_string()),
        };

        self.espera(Token::Em)?;

        let colecao = match self.avancar() {
            Some(Token::Identificador(c)) => c.clone(),
            _ => return Err("Esperado coleção em 'para cada'".to_string()),
        };

        let corpo = self.parse_bloco()?;

        Ok(Comando::ParaCada { var, colecao, corpo })
    }

    fn parse_expressao(&mut self) -> Result<Expr, String> {
        self.parse_igualdade()
    }

    fn parse_igualdade(&mut self) -> Result<Expr, String> {
        let mut esq = self.parse_comparacao()?;

        while let Some(tok) = self.atual() {
            match tok {
                Token::Igual => {
                    self.avancar();
                    let dir = self.parse_comparacao()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::Igual,
                        dir: Box::new(dir),
                    };
                }
                Token::Diferente => {
                    self.avancar();
                    let dir = self.parse_comparacao()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::Diferente,
                        dir: Box::new(dir),
                    };
                }
                _ => break,
            }
        }

        Ok(esq)
    }

    fn parse_comparacao(&mut self) -> Result<Expr, String> {
        let mut esq = self.parse_adicao()?;

        while let Some(tok) = self.atual() {
            match tok {
                Token::Menor => {
                    self.avancar();
                    let dir = self.parse_adicao()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::Menor,
                        dir: Box::new(dir),
                    };
                }
                Token::Maior => {
                    self.avancar();
                    let dir = self.parse_adicao()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::Maior,
                        dir: Box::new(dir),
                    };
                }
                Token::MenorIgual => {
                    self.avancar();
                    let dir = self.parse_adicao()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::MenorIgual,
                        dir: Box::new(dir),
                    };
                }
                Token::MaiorIgual => {
                    self.avancar();
                    let dir = self.parse_adicao()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::MaiorIgual,
                        dir: Box::new(dir),
                    };
                }
                _ => break,
            }
        }

        Ok(esq)
    }

    fn parse_adicao(&mut self) -> Result<Expr, String> {
        let mut esq = self.parse_multiplicacao()?;

        while let Some(tok) = self.atual() {
            match tok {
                Token::Mais => {
                    self.avancar();
                    let dir = self.parse_multiplicacao()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::Soma,
                        dir: Box::new(dir),
                    };
                }
                Token::Menos => {
                    self.avancar();
                    let dir = self.parse_multiplicacao()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::Subtracao,
                        dir: Box::new(dir),
                    };
                }
                _ => break,
            }
        }

        Ok(esq)
    }

    fn parse_multiplicacao(&mut self) -> Result<Expr, String> {
        let mut esq = self.parse_primario()?;

        while let Some(tok) = self.atual() {
            match tok {
                Token::Vezes => {
                    self.avancar();
                    let dir = self.parse_primario()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::Multiplicacao,
                        dir: Box::new(dir),
                    };
                }
                Token::Dividido => {
                    self.avancar();
                    let dir = self.parse_primario()?;
                    esq = Expr::Binaria {
                        esq: Box::new(esq),
                        op: Operador::Divisao,
                        dir: Box::new(dir),
                    };
                }
                _ => break,
            }
        }

        Ok(esq)
    }

    fn parse_primario(&mut self) -> Result<Expr, String> {
        let tok = self.avancar_clonado().ok_or("Expressão esperada")?;
        match tok {
            Token::InteiroLit(n) => Ok(Expr::Literal(Literal::Inteiro(n))),
            Token::DecimalLit(n) => Ok(Expr::Literal(Literal::Decimal(n, 2))),
            Token::TextoLit(s) => Ok(Expr::Literal(Literal::Texto(s))),
            Token::Identificador(nome) => {
                let is_chamada = if let Some(Token::ParenEsq) = self.atual() { true } else { false };
                if is_chamada {
                    self.avancar();
                    let mut args = Vec::new();
                    while let Some(t) = self.atual() {
                        if let Token::ParenDir = t {
                            break;
                        }
                        args.push(self.parse_expressao()?);
                        if let Some(Token::Virgula) = self.atual() {
                            self.avancar();
                        }
                    }
                    self.espera(Token::ParenDir)?;
                    Ok(Expr::Chamada { funcao: nome, args })
                } else {
                    Ok(Expr::Identificador(nome))
                }
            }
            _ => Err(format!("Expressão esperada, encontrado {:?}", tok)),
        }
    }
}

pub fn parse(input: &str) -> Result<Vec<Declaracao>, String> {
    let mut lexer = Token::lexer(input);
    let mut tokens = Vec::new();
    while let Some(tok) = lexer.next() {
        match tok {
            Ok(t) => tokens.push(t),
            Err(_) => return Err("Erro léxico".to_string()),
        }
    }
    let mut parser = Parser::new(tokens);
    let mut declaracoes = Vec::new();
    while parser.atual().is_some() {
        declaracoes.push(parser.parse_declaracao()?);
    }
    Ok(declaracoes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_icms_simples() {
        let codigo = r#"
        funcao calcular_quota(receita: Decimal, populacao: Inteiro) -> Decimal
            verificavel_mathematicamente
            com_precisao(decimais: 2)
        {
            retorne receita * populacao / 1000000;
        }
        "#;
        let resultado = parse(codigo);
        assert!(resultado.is_ok(), "Erro: {:?}", resultado.err());
        let decls = resultado.unwrap();
        assert_eq!(decls.len(), 1);
    }

    #[test]
    fn test_parse_processo() {
        let codigo = r#"
        processo reparticao(valor: Decimal) -> Decimal
            auditavel_por("TCU")
        {
            etapa calculo {
                retorne valor * 0.25;
            }
        }
        "#;
        let resultado = parse(codigo);
        assert!(resultado.is_ok(), "Erro: {:?}", resultado.err());
    }
}
