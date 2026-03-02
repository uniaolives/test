#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "VajraCircuitBreaker.generated.h"

UENUM(BlueprintType)
enum class EByzantineFault : uint8
{
    GRAVITY_INVERSION,
    TEMPORAL_DILATION,
    QUANTUM_DECOHERENCE,
    CAUSALITY_VIOLATION,
    SOCIAL_CONTAGION
};

USTRUCT(BlueprintType)
struct FVajraStateSnapshot
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float Phi = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    float Entropy = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    FString StateHash;

    UPROPERTY(BlueprintReadOnly)
    int64 Timestamp;

    UPROPERTY(BlueprintReadOnly)
    bool bIsCoherent = true;

    UPROPERTY(BlueprintReadOnly)
    FString AnomalyType;
};

USTRUCT(BlueprintType)
struct FTMRInstance
{
    GENERATED_BODY()

    UPROPERTY()
    int32 InstanceID = 0;

    UPROPERTY()
    FVector AgentPosition;

    UPROPERTY()
    FVector GravityVector;

    UPROPERTY()
    FString PhysicsHash;

    UPROPERTY()
    bool bIsHealthy = true;

    UPROPERTY()
    float LastUpdateTime = 0.0f;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnHardFreeze, const FString&, FreezeReason);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnByzantineDetected, int32, FaultyInstance, const FString&, Details);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnPhiChanged, float, NewPhiValue);

UCLASS(ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class VAJRACIRCUITBREAKER_API UVajraCircuitBreaker : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;

    // Core System Initialization
    UFUNCTION(BlueprintCallable, Category = "Crux86")
    void InitializeTMRSystem(const FString& SatoshiSeed, float InitialPhi = 0.72f);

    // Physics State Submission
    UFUNCTION(BlueprintCallable, Category = "Crux86")
    void SubmitPhysicsState(int32 InstanceID, const FTransform& AgentTransform,
                           const FVector& CurrentGravity, const FVector& LinearVelocity);

    // Byzantine Fault Injection (for testing)
    UFUNCTION(BlueprintCallable, Category = "Crux86|Testing")
    void InjectByzantineFault(EByzantineFault FaultType, int32 TargetInstance,
                              const FString& Payload = TEXT(""));

    // Ω-Prevention Core Functions
    UFUNCTION(BlueprintCallable, Category = "Crux86|Security")
    FVajraStateSnapshot ValidateTMRConsensus();

    UFUNCTION(BlueprintCallable, Category = "Crux86|Security")
    void TriggerHardFreeze(const FString& Reason, float CurrentPhi = 0.80f);

    // Mesh-Neuron Integration
    UFUNCTION(BlueprintCallable, Category = "Crux86|Network")
    void ReportToMeshNeuron(const FVajraStateSnapshot& Snapshot);

    // KARNAK Sealing
    UFUNCTION(BlueprintCallable, Category = "Crux86|Persistence")
    void SealStateToKarnak(const FString& SealType, const FVajraStateSnapshot& State);

    // Empathy Protocol (Dor do Boto)
    UFUNCTION(BlueprintCallable, Category = "Crux86|Social")
    void UpdateSocialStress(float StressLevel, float CortisolDamping = 0.69f);

    // Blueprint Events
    UPROPERTY(BlueprintAssignable, Category = "Crux86|Events")
    FOnHardFreeze OnHardFreeze;

    UPROPERTY(BlueprintAssignable, Category = "Crux86|Events")
    FOnByzantineDetected OnByzantineDetected;

    UPROPERTY(BlueprintAssignable, Category = "Crux86|Events")
    FOnPhiChanged OnPhiChanged;

protected:
    // TMR Management
    UPROPERTY()
    TArray<FTMRInstance> TMRInstances;

    // System State
    UPROPERTY()
    float CurrentPhi = 0.72f;

    UPROPERTY()
    float PhiCriticalThreshold = 0.80f;

    UPROPERTY()
    float PhiWarningThreshold = 0.72f;

    UPROPERTY()
    FString SatoshiSeedHash;

    UPROPERTY()
    bool bHardFreezeActive = false;

    UPROPERTY()
    TArray<FVajraStateSnapshot> StateHistory;

private:
    // Internal Functions
    FString GenerateBlake3Hash(const FString& Input);
    float CalculateVariance(const TArray<FVector>& Positions);
    void IsolateFaultyInstance(int32 InstanceID);
    void NotifySASCathedral(const FString& EventType, const FVajraStateSnapshot& State);
    void ApplyEmpathyDamping(float StressLevel, float DampingFactor);
};
