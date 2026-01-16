CC = gcc
CFLAGS = -Wall -Wextra -I.
SRCS = constitutional_bitchat.c \
       constitutional_bitchat_privacy.c \
       constitutional_bitchat_integration.c \
       constitutional_web3_improvement.c \
       constitutional_base.c \
       constitutional_lfs_mock.c
OBJS = $(SRCS:.c=.o)
LIB = libbitchat.a
TEST_SRC = test_bitchat.c
TEST_EXE = test_bitchat

all: $(LIB) $(TEST_EXE)

$(LIB): $(OBJS)
	ar rcs $@ $(OBJS)

$(TEST_EXE): $(TEST_SRC) $(LIB)
	$(CC) $(CFLAGS) $< -L. -lbitchat -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB) $(TEST_EXE) LFS_AUDIT_LOG_CYCLE_1.md

.PHONY: all clean
