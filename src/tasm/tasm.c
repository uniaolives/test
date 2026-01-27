#include "tasm.h"

#define MAX_PROGRAM_SIZE 1024

void push_program(Inst program[], int *program_size, Inst value){
    assert(*program_size < MAX_PROGRAM_SIZE && "Program size cannot exceed max program size\n");
    program[(*program_size)++] = value;
}

Inst pop_program(Inst program[], int program_size){
    assert(program_size > 0 && "STACK UNDERFLOW\n");
    return program[--program_size];
}

size_t length_of_list(ParseList *head){
    ParseList *tmp = head;
    size_t result = 1;
    while(tmp->next != NULL){
        result += 1;
        tmp = tmp->next;
    }
    return result;
}

char *remove_first_character(char *str){
    int length = strlen(str);
    char *result = malloc(sizeof(char) * length);
    for(int i = 0; i < length - 1; i++){
        result[i] = str[i + 1];
    }
    result[length - 1] = '\0';
    return result;
}

size_t get_register_index(ParseList *head){
    size_t result = (size_t)atoi(remove_first_character(head->value.text));
    if(result >= AMOUNT_OF_REGISTERS){
        fprintf(stderr, "error: register index is too great\n");
        exit(1);
    }
    return result;
}

static size_t str_stack_size = 0;

Inst *generate_instructions(ParseList *head, int *program_size, Str_Stack *str_stack, size_t *entrypoint, bool *has_entrypoint){
    Inst *program = malloc(sizeof(Inst) * length_of_list(head));
    Inst_Set insts[TYPE_COUNT] = {0};
    insts[TYPE_NOP] = INST_NOP; insts[TYPE_PUSH] = INST_PUSH; insts[TYPE_PUSH_STR] = INST_PUSH_STR;
    insts[TYPE_MOV] = INST_MOV; insts[TYPE_REF] = INST_REF; insts[TYPE_DEREF] = INST_DEREF;
    insts[TYPE_ALLOC] = INST_ALLOC; insts[TYPE_DEALLOC] = INST_DEALLOC; insts[TYPE_WRITE] = INST_WRITE;
    insts[TYPE_READ] = INST_READ; insts[TYPE_POP] = INST_POP; insts[TYPE_DUP] = INST_DUP;
    insts[TYPE_INDUP] = INST_INDUP; insts[TYPE_SWAP] = INST_SWAP; insts[TYPE_INSWAP] = INST_INSWAP;
    insts[TYPE_ADD] = INST_ADD; insts[TYPE_SUB] = INST_SUB; insts[TYPE_MUL] = INST_MUL;
    insts[TYPE_DIV] = INST_DIV; insts[TYPE_MOD] = INST_MOD; insts[TYPE_AND] = INST_AND;
    insts[TYPE_OR] = INST_OR; insts[TYPE_ADD_F] = INST_ADD_F; insts[TYPE_SUB_F] = INST_SUB_F;
    insts[TYPE_MUL_F] = INST_MUL_F; insts[TYPE_DIV_F] = INST_DIV_F; insts[TYPE_MOD_F] = INST_MOD_F;
    insts[TYPE_CMPE] = INST_CMPE; insts[TYPE_CMPNE] = INST_CMPNE; insts[TYPE_CMPG] = INST_CMPG;
    insts[TYPE_CMPL] = INST_CMPL; insts[TYPE_CMPGE] = INST_CMPGE; insts[TYPE_CMPLE] = INST_CMPLE;
    insts[TYPE_ITOF] = INST_ITOF; insts[TYPE_FTOI] = INST_FTOI; insts[TYPE_ITOC] = INST_ITOC;
    insts[TYPE_TOI] = INST_TOI; insts[TYPE_TOF] = INST_TOF; insts[TYPE_TOC] = INST_TOC;
    insts[TYPE_TOVP] = INST_TOVP; insts[TYPE_CALL] = INST_CALL; insts[TYPE_RET] = INST_RET;
    insts[TYPE_JMP] = INST_JMP; insts[TYPE_ZJMP] = INST_ZJMP; insts[TYPE_NZJMP] = INST_NZJMP;
    insts[TYPE_PRINT] = INST_PRINT; insts[TYPE_NATIVE] = INST_NATIVE; insts[TYPE_ENTRYPOINT] = INST_ENTRYPOINT;
    insts[TYPE_LOAD_LIB] = INST_LOAD_LIBRARY; insts[TYPE_SS] = INST_SS; insts[TYPE_HALT] = INST_HALT;
    insts[TYPE_GET_STR] = INST_GET_STR; insts[TYPE_DUP_STR] = INST_DUP_STR;
    insts[TYPE_STRLEN] = INST_STRLEN; insts[TYPE_INDEX] = INST_INDEX;

    while(head != NULL){
        assert(head->value.type != TYPE_NONE && "Value should not be none\n");
        assert(head->value.type < (TokenType)TYPE_COUNT && "Incorrect value\n");
        Inst *instruction = malloc(sizeof(Inst));
        instruction->type = insts[head->value.type];
        if(
                head->value.type == TYPE_CALL || head->value.type == TYPE_NATIVE || 
                head->value.type == TYPE_JMP || head->value.type == TYPE_ZJMP || 
                head->value.type == TYPE_NZJMP
        ){
            head = head->next;
            instruction->value.as_int = atoi(head->value.text);
            instruction->data_type = INT_TYPE;
        }

        if(head->value.type == TYPE_ENTRYPOINT){
            instruction->type = INST_NOP;
            head = head->next;
            if(!*has_entrypoint){
                *entrypoint = (size_t)atoi(head->value.text);
                *has_entrypoint = true;
            } else {
                fprintf(stderr, "error: cannot define entrypoint more than once\n");
                exit(1);
            }
        }


        if(head->value.type == TYPE_PUSH || head->value.type == TYPE_INSWAP || head->value.type == TYPE_INDUP || head->value.type == TYPE_GET_STR){
            head = head->next;
            if(head->value.type == TYPE_INT){
                instruction->value.as_int = atoi(head->value.text);
                instruction->data_type = INT_TYPE;
            } else if(head->value.type == TYPE_FLOAT){
                instruction->value.as_float = atof(head->value.text);
                instruction->data_type = FLOAT_TYPE;
            } else if(head->value.type == TYPE_CHAR){
                instruction->value.as_char = head->value.text[0];
                instruction->data_type = CHAR_TYPE;
            } else if(check_if_register(head->value.type)){
                instruction->register_index = get_register_index(head);             
                instruction->data_type = REGISTER_TYPE;
            } else {
                assert(false && "you should not be here\n");
            }
        }

        if(head->value.type == TYPE_MOV){
            head = head->next;
            instruction->register_index = get_register_index(head);             
            head = head->next;
            if(head->value.type == TYPE_INT){
                instruction->value.as_int = atoi(head->value.text);
                instruction->data_type = INT_TYPE;
            } else if(head->value.type == TYPE_FLOAT){
                instruction->value.as_float = atof(head->value.text);
                instruction->data_type = FLOAT_TYPE;
            } else if(head->value.type == TYPE_CHAR){
                instruction->value.as_char = head->value.text[0];
                instruction->data_type = CHAR_TYPE;
            } else if(head->value.type == TYPE_TOP){
                instruction->data_type = TOP_TYPE;
            } else {
                assert(false && "you should not be here\n");
            }
        }

        if(head->value.type == TYPE_PUSH_STR){
            head = head->next;
            if(head->value.type != TYPE_STRING){
                assert(false && "why are you here\n");
            }
            instruction->type = INST_PUSH_STR;
            instruction->value.as_int = str_stack_size;
            instruction->data_type = INT_TYPE;
            size_t str_s = strlen(head->value.text)+1;
            String_View sv;
            sv.len = str_s;
            sv.data = malloc(sizeof(char)*str_s);
            strncpy((char*)sv.data, head->value.text, str_s);
            DA_APPEND(str_stack, sv);
            str_stack_size++;
        }

        push_program(program, program_size, *instruction);
        free(instruction);
        head = head->next;
    }
    return program;
}

char *chop_file_by_dot(char *file_name){
    int index;
    char *result = malloc(sizeof(char) * 64);
    for(index = 0; file_name[index] != '.' && file_name[index] != '\0'; index++){
        result[index] = file_name[index];
    }
    snprintf(result + index, 5, ".tim");
    result = realloc(result, sizeof(char) * index);
    return result;
}

int main(int argc, char *argv[]){
    if(argc < 2){
        fprintf(stderr, "Usage: %s <file_name.tasm>\n", argv[0]);
        exit(1);
    }

    char *file_name = argv[1];
    char *output_file = chop_file_by_dot(file_name);
    Lexer lex = lexer(file_name);
    ParseList list = parser(lex);
    //print_list(&list);
    int program_size = 0;
    Machine machine;
    size_t entrypoint = 0;
    bool has_entrypoint = false;
    machine.str_stack.data = NULL;
    machine.str_stack.count = 0;
    machine.str_stack.capacity = 0;
    Inst *program = generate_instructions(&list, &program_size, &machine.str_stack, &entrypoint, &has_entrypoint);
    machine.instructions.data = program;
    machine.instructions.count = program_size;
    machine.program_size = program_size;
    machine.entrypoint = entrypoint;
    machine.has_entrypoint = has_entrypoint;
    write_program_to_file(&machine, output_file);
    return 0;
}
