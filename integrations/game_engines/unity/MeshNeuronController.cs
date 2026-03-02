using System;
using System.Collections;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace Crux86
{
    [System.Serializable]
    public class TMRState
    {
        public int instanceId;
        public Vector3 position;
        public Vector3 gravity;
        public string hash;
        public long timestamp;
        public bool isHealthy = true;
    }

    [System.Serializable]
    public class VajraSnapshot
    {
        public float phi;
        public float entropy;
        public string stateHash;
        public long timestamp;
        public bool isCoherent;
        public string anomalyType;
    }

    [System.Serializable]
    public class KarnakSeal
    {
        public string seal_id;
        public long timestamp;
        public string type;
        public string content_hash;
        public string algorithm;
        public string satoshi_anchor;
    }

    public class MeshNeuronController : MonoBehaviour
    {
        [Header("Crux86 Configuration")]
        [SerializeField] private string satoshiSeed = "0xbd36332890d15e2f360bb65775374b462b";
        [SerializeField] private float phiThreshold = 0.72f;
        [SerializeField] private float phiCritical = 0.80f;

        [Header("TMR Configuration")]
        [SerializeField] private int tmrInstanceCount = 3;
        [SerializeField] private float consensusVarianceThreshold = 0.000032f;

        [Header("Network Endpoints")]
        [SerializeField] private string meshNeuronEndpoint = "http://localhost:3030";
        [SerializeField] private string karnakEndpoint = "http://localhost:9091";
        [SerializeField] private string sascEndpoint = "http://localhost:12800";

        private List<TMRState> tmrInstances = new List<TMRState>();
        private bool hardFreezeActive = false;
        private float currentPhi = 0.72f;
        private float cortisolLevel = 0f;

        public event Action<float> OnPhiChanged;
        public event Action<string> OnHardFreeze;
        public event Action<int, string> OnByzantineDetected;

        void Start()
        {
            InitializeTMR();
            StartCoroutine(MonitoringCoroutine());
        }

        void InitializeTMR()
        {
            tmrInstances.Clear();

            for (int i = 0; i < tmrInstanceCount; i++)
            {
                tmrInstances.Add(new TMRState
                {
                    instanceId = i + 1,
                    gravity = Physics.gravity,
                    timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    isHealthy = true
                });
            }

            Debug.Log($"[Mesh-Neuron] Initialized {tmrInstanceCount} TMR instances");
            Debug.Log($"[Mesh-Neuron] Satoshi seed: {satoshiSeed.Substring(0, 24)}...");
        }

        public void SubmitPhysicsState(int instanceId, Vector3 position, Vector3 gravity)
        {
            if (hardFreezeActive) return;

            if (instanceId < 1 || instanceId > tmrInstances.Count)
            {
                Debug.LogError($"[Mesh-Neuron] Invalid instance ID: {instanceId}");
                return;
            }

            var instance = tmrInstances[instanceId - 1];
            instance.position = position;
            instance.gravity = gravity;
            instance.timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            instance.hash = ComputeBlake3Hash($"{position}|{gravity}|{instance.timestamp}");
        }

        public VajraSnapshot ValidateConsensus()
        {
            var snapshot = new VajraSnapshot
            {
                timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                isCoherent = true
            };

            if (hardFreezeActive)
            {
                snapshot.anomalyType = "HARD_FREEZE_ACTIVE";
                return snapshot;
            }

            // Collect data from healthy instances
            var positions = new List<Vector3>();
            var gravities = new List<Vector3>();
            var hashes = new List<string>();

            foreach (var instance in tmrInstances)
            {
                if (instance.isHealthy)
                {
                    positions.Add(instance.position);
                    gravities.Add(instance.gravity);
                    hashes.Add(instance.hash);
                }
            }

            // Calculate variances
            float positionVariance = CalculateVariance(positions);
            float gravityVariance = CalculateVariance(gravities);

            // Calculate Phi
            snapshot.phi = 1.0f - Mathf.Sqrt(positionVariance + gravityVariance);
            snapshot.entropy = Mathf.Log(1.0f + positionVariance * 1000f);

            // Check hash consensus
            var hashCounts = new Dictionary<string, int>();
            foreach (var hash in hashes)
            {
                if (hashCounts.ContainsKey(hash))
                    hashCounts[hash]++;
                else
                    hashCounts[hash] = 1;
            }

            bool consensus = false;
            string majorityHash = "";
            foreach (var kvp in hashCounts)
            {
                if (kvp.Value >= 2) // 2/3 consensus
                {
                    consensus = true;
                    majorityHash = kvp.Key;
                    break;
                }
            }

            // Detect Byzantine faults
            if (!consensus || positionVariance > consensusVarianceThreshold)
            {
                for (int i = 0; i < tmrInstances.Count; i++)
                {
                    if (tmrInstances[i].isHealthy && tmrInstances[i].hash != majorityHash)
                    {
                        IsolateInstance(i + 1);
                        snapshot.anomalyType = $"BYZANTINE_INSTANCE_{i + 1}";
                        snapshot.isCoherent = false;

                        OnByzantineDetected?.Invoke(i + 1, "TMR consensus failure");
                        break;
                    }
                }
            }

            // Check Phi thresholds
            if (snapshot.phi >= phiCritical)
            {
                TriggerHardFreeze($"PHI_CRITICAL_{snapshot.phi}", snapshot.phi);
                snapshot.anomalyType = "PHI_CRITICAL";
                snapshot.isCoherent = false;
            }
            else if (snapshot.phi >= phiThreshold)
            {
                Debug.LogWarning($"[Mesh-Neuron] Phi warning: {snapshot.phi}");
                snapshot.anomalyType = "PHI_WARNING";
            }

            // Update current Phi
            if (Mathf.Abs(currentPhi - snapshot.phi) > 0.01f)
            {
                currentPhi = snapshot.phi;
                OnPhiChanged?.Invoke(currentPhi);
            }

            snapshot.stateHash = ComputeBlake3Hash($"{snapshot.phi}|{snapshot.timestamp}");
            return snapshot;
        }

        void TriggerHardFreeze(string reason, float phi)
        {
            if (hardFreezeActive) return;

            hardFreezeActive = true;
            Debug.LogError($"[VAJRA HARD FREEZE] {reason} (Φ={phi})");

            // 1. Freeze time
            Time.timeScale = 0f;

            // 2. Seal to KARNAK
            StartCoroutine(SealToKarnak(reason, phi));

            // 3. Notify SASC
            StartCoroutine(NotifySASC(reason, phi));

            // 4. Trigger event
            OnHardFreeze?.Invoke(reason);
        }

        IEnumerator SealToKarnak(string reason, float phi)
        {
            var seal = new KarnakSeal
            {
                seal_id = Guid.NewGuid().ToString(),
                timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                type = "hard_freeze",
                content_hash = ComputeBlake3Hash($"{reason}|{phi}|{Time.time}"),
                algorithm = "blake2b_256",
                satoshi_anchor = satoshiSeed
            };

            string json = JsonUtility.ToJson(seal);

            using (UnityWebRequest request = UnityWebRequest.Post(karnakEndpoint + "/seal", json, "application/json"))
            {
                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    Debug.Log("[KARNAK] State sealed successfully");
                }
                else
                {
                    Debug.LogError("[KARNAK] Seal failed: " + request.error);
                }
            }
        }

        IEnumerator NotifySASC(string reason, float phi)
        {
            string json = $"{{\"event\":\"hard_freeze\",\"reason\":\"{reason}\",\"phi\":{phi},\"timestamp\":{DateTimeOffset.UtcNow.ToUnixTimeSeconds()}}}";

            using (UnityWebRequest request = UnityWebRequest.Post(sascEndpoint + "/v1/emergency/freeze", json, "application/json"))
            {
                yield return request.SendWebRequest();
            }
        }

        float CalculateVariance(List<Vector3> vectors)
        {
            if (vectors.Count < 2) return 0f;

            // Calculate mean
            Vector3 mean = Vector3.zero;
            foreach (var vec in vectors) mean += vec;
            mean /= vectors.Count;

            // Calculate variance
            float variance = 0f;
            foreach (var vec in vectors)
            {
                float distance = Vector3.Distance(vec, mean);
                variance += distance * distance;
            }
            variance /= vectors.Count;

            return variance;
        }

        void IsolateInstance(int instanceId)
        {
            if (instanceId < 1 || instanceId > tmrInstances.Count) return;

            tmrInstances[instanceId - 1].isHealthy = false;
            Debug.LogWarning($"[Mesh-Neuron] Isolated instance {instanceId}");

            // Check if we still have quorum
            int healthyCount = 0;
            foreach (var instance in tmrInstances)
                if (instance.isHealthy) healthyCount++;

            if (healthyCount < 2)
            {
                TriggerHardFreeze("TMR_QUORUM_LOST", currentPhi);
            }
        }

        string ComputeBlake3Hash(string input)
        {
            // In production, use actual BLAKE3 library
            using (SHA256 sha256 = SHA256.Create())
            {
                byte[] bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(input + satoshiSeed));
                return BitConverter.ToString(bytes).Replace("-", "").ToLower();
            }
        }

        public void UpdateSocialStress(float stress, float damping = 0.69f)
        {
            cortisolLevel = stress * (1f - damping);

            if (cortisolLevel > 0.3f)
            {
                Debug.LogWarning($"[Dor do Boto] High social stress: {cortisolLevel}");

                // Apply stress reduction measures
                ReduceSocialComplexity(0.5f);
            }
        }

        void ReduceSocialComplexity(float factor)
        {
            // Reduce NPC count
            var npcs = GameObject.FindGameObjectsWithTag("NPC");
            foreach (var npc in npcs)
            {
                if (UnityEngine.Random.value > factor)
                    Destroy(npc);
            }

            // Reduce particle systems
            var particles = FindObjectsOfType<ParticleSystem>();
            foreach (var ps in particles)
            {
                var emission = ps.emission;
                emission.rateOverTime *= factor;
            }
        }

        IEnumerator MonitoringCoroutine()
        {
            while (true)
            {
                ValidateConsensus();
                yield return new WaitForSeconds(0.1f); // 10Hz monitoring
            }
        }
    }
}
