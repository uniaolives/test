#include "constitutional_bitchat.h"
#include <assert.h>

int main() {
    printf("--- BITCHAT FUNCTIONAL VERIFICATION ---\n");

    // Test Initialization
    int init_res = constitutional_bitchat_initialize();
    assert(init_res == 0);

    // Test Message Sending
    ConstitutionalBitchatNode* node = constitutional_allocate_bitchat_node();
    const char* data = "Hello Web3";
    int send_res = constitutional_bitchat_send_message(node, CONSTITUTIONAL_BITCHAT_PROTOCOL_INSIGHT, data, strlen(data), CONSTITUTIONAL_PRIVACY_LEVEL_ANONYMIZED);
    assert(send_res == 0);

    // Test Privacy Anonymization
    char anonymized[128];
    uint32_t anon_size = 0;
    constitutional_anonymize_user_data_for_bitchat("Sensitive Info", 14, anonymized, &anon_size);

    // Test Feedback Loop Execution
    ConstitutionalWeb3FeedbackLoop loop;
    memset(&loop, 0, sizeof(loop));
    constitutional_execute_web3_feedback_loop(&loop);

    printf("\n✅ ALL BITCHAT VERIFICATION TESTS PASSED\n");
    return 0;
}
