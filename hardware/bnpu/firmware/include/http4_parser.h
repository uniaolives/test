// http4_parser.h
#ifndef HTTP4_PARSER_H
#define HTTP4_PARSER_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    HTTP4_OBSERVE,
    HTTP4_EMIT,
    HTTP4_ENTANGLE,
    HTTP4_COLLAPSE,
    HTTP4_QUANTIZE,
    HTTP4_PROPAGATE,
    HTTP4_RECALL,
    HTTP4_REANIMATE
} Http4Method;

typedef struct {
    Http4Method method;
    uint64_t target_epoch;
    // other fields...
} Http4Request;

void http4_init(void);

#endif
