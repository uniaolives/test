import * as TaskManager from 'expo-task-manager';
import * as BackgroundFetch from 'expo-background-fetch';
import { useArkheStore } from '../store/arkheStore';

const BACKGROUND_SYNC_TASK = 'arkhe-background-sync';

TaskManager.defineTask(BACKGROUND_SYNC_TASK, async () => {
  try {
    const store = useArkheStore.getState();

    // 1. Check quantum coherence
    const qkdActive = store.qkdSessions.filter(s => s.status === 'active').length;

    // 2. Monitor drone battery levels
    const lowBatteryDrones = store.drones.filter(d => d.battery_level < 20);

    // 3. Sync system logs to blockchain if threshold reached
    if (store.systemLogs.length > 10) {
      const latestLog = store.systemLogs[store.systemLogs.length - 1];
      if (!latestLog.txHash && store.contracts['ArkheLedger']) {
        await store.logToBlockchain({
          device: latestLog.device,
          cpuLoad: latestLog.cpuLoad,
          memoryUsage: latestLog.memoryUsage,
        });
      }
    }

    // 4. Recalculate global coherence
    store.calculateGlobalCoherence();

    return BackgroundFetch.BackgroundFetchResult.NewData;
  } catch (error) {
    console.error('Background sync failed:', error);
    return BackgroundFetch.BackgroundFetchResult.Failed;
  }
});

export async function registerBackgroundSync() {
  try {
    await BackgroundFetch.registerTaskAsync(BACKGROUND_SYNC_TASK, {
      minimumInterval: 60, // 1 minute
      stopOnTerminate: false,
      startOnBoot: true,
    });
    console.log('Background sync registered');
  } catch (error) {
    console.error('Failed to register background sync:', error);
  }
}
