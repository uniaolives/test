#ifndef TASM_H
#define TASM_H

#include <stdio.h>
#include <stdlib.h>

#include "tasmlexer.h"
#include "tasmparser.h"
#include "tim.h"

void push_program(Inst program[], int *program_size, Inst value);
Inst pop_program(Inst program[], int program_size);
size_t length_of_list(ParseList *head);
Inst *generate_instructions(ParseList *head, int *program_size, Str_Stack *str_stack, size_t *entrypoint, bool *has_entrypoint);
char *chop_file_by_dot(char *file_name);
int main(int argc, char *argv[]);

#endif
