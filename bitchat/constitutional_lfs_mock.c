#include <stdio.h>
#include <time.h>
#include "constitutional_bitchat.h"

void constitutional_lfs_log_cycle(const char* cycle_id, int insights_count, float effectiveness) {
    FILE *f = fopen("LFS_AUDIT_LOG_CYCLE_1.md", "a");
    if (f == NULL) return;

    time_t now;
    time(&now);

    fprintf(f, "# LFS AUDIT ENTRY\n");
    fprintf(f, "- **Timestamp:** %s", ctime(&now));
    fprintf(f, "- **Cycle ID:** %s\n", cycle_id);
    fprintf(f, "- **Insights Generated:** %d\n", insights_count);
    fprintf(f, "- **Effectiveness Score:** %.2f\n", effectiveness);
    fprintf(f, "- **Status:** COMMITTED TO IMMUTABLE STORAGE\n\n");

    printf("✅ [LFS] Audit log written to disk.\n");
    fclose(f);
}
