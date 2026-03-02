// Exemplo de Repartição de ICMS
funcao calcular_icms_estadual(valor_total: Decimal, aliquota: Decimal) -> Decimal
    verificavel_mathematicamente
    com_precisao(decimais: 2)
{
    retorne valor_total * aliquota;
}

processo reparticao_icms(arrecadacao: Decimal) -> Decimal
    auditavel_por("Tribunal de Contas")
{
    etapa calculo {
        retorne arrecadacao * 0.25;
    }
}
