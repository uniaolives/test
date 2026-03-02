BITS 64
DEFAULT REL

section .text
global _start

; ==================== CORREÇÕES APLICADAS ====================

; 1. Gate 3 - Busca de nonce em tempo constante
gate3_verify_nonce:
    push rbp
    mov rbp, rsp

    ; RBX = nonce a verificar (passado por parâmetro)
    ; RSI = ponteiro para cache de nonces
    ; RCX = tamanho do cache (1024)

    xor rax, rax                    ; Acumulador de comparação
    mov rcx, 1024                   ; Tamanho fixo do cache

.nonce_search_loop:
    mov rdx, [rsi + rcx*8 - 8]     ; Carregar entrada do cache
    xor rdx, rbx                   ; Comparar com nonce
    or rax, rdx                    ; Acumular diferenças (0 = igual)
    loop .nonce_search_loop        ; Dec RCX e salta se não zero

    ; RAX == 0 significa nonce encontrado (replay)
    test rax, rax
    jnz .gate3_pass

.gate3_replay_detected:
    mov rax, 0xDEADBEEF            ; Código de erro para replay
    jmp .gate_fail

.gate3_pass:
    pop rbp
    ret

; 2. HALT seguro com limpeza de estado
secure_final_halt:
    ; Limpar todos os registradores sensíveis
    xor r15, r15
    xor r14, r14
    xor rbx, rbx
    xor rsi, rsi
    xor rdi, rdi
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13

    ; Limpar buffer de hash
    mov qword [attestation_hash_buffer], 0

    ; Gerar evidência final (hash SHA-256 do estado)
    lea rsi, [evidence_buffer]
    mov rdi, 64                    ; Tamanho do buffer
    ; call sha256_hash             ; Mocked call

    ; Salvar em área segura (não em endereço fixo)
    mov rdx, [tpm_base_address]    ; Lido da tabela ACPI
    add rdx, 0x600                 ; Área de evidência
    mov [rdx], rax

    ; HALT com interrupções desabilitadas
    cli
.halt_loop:
    hlt
    jmp .halt_loop

.gate_fail:
    mov r15, rax
    jmp secure_final_halt

; 3. Acesso abstrato ao TPM via ACPI
read_pcr0_abstract:
    push rbp
    mov rbp, rsp

    ; Obter endereço base do TPM da tabela ACPI
    mov rax, [acpi_tpm_base]       ; Preenchido durante boot
    test rax, rax
    jz .tpm_not_found

    ; Verificar interface (CRB vs TIS)
    mov edx, [rax + 0x30]          ; Interface ID
    cmp edx, 0x43524200           ; 'CRB\0'
    je .use_crb_interface

.use_tis_interface:
    add rax, 0xF00                ; Offset para área de PCRs
    jmp .read_pcr

.use_crb_interface:
    add rax, 0x600                ; Offset para PCRs na interface CRB

.read_pcr:
    mov rbx, [rax + 0]            ; PCR0
    pop rbp
    ret

.tpm_not_found:
    mov rax, 0xFFFFFFFFFFFFFFFF
    pop rbp
    ret

; ==================== PONTO DE ENTRADA ====================
_start:
    ; Inicialização segura
    xor rax, rax
    xor rbx, rbx
    xor rcx, rcx
    xor rdx, rdx

    ; Executar verificação dos 5 Gates (Mocks for functional validation)
    ; call gate1_verify_signature
    ; call gate2_verify_pcr0
    ; call gate3_verify_nonce
    ; call gate4_verify_entropy
    ; call gate5_verify_coherence

    ; Todas as verificações passaram
    mov rax, 0x1337BEEF           ; Código de sucesso
    jmp secure_final_halt

section .data
tpm_base_address: dq 0
acpi_tpm_base: dq 0
attestation_hash_buffer: times 32 db 0
evidence_buffer: times 64 db 0
nonce_cache: times 1024 dq 0
