#include "VajraCircuitBreaker.h"
#include "Misc/SecureHash.h"
#include "HAL/PlatformProcess.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "JsonObjectConverter.h"
#include "Chaos/ChaosInterface.h"
#include "Engine/World.h"
#include "Engine/Engine.h"

void UVajraCircuitBreaker::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    // Load Satoshi seed from environment or config
    SatoshiSeedHash = FPlatformMisc::GetEnvironmentVariable(TEXT("SATOSHI_SEED"));
    if (SatoshiSeedHash.IsEmpty())
    {
        SatoshiSeedHash = TEXT("0xbd36332890d15e2f360bb65775374b462b");
    }

    UE_LOG(LogTemp, Log, TEXT("Vajra Circuit Breaker Initialized"));
}

void UVajraCircuitBreaker::Deinitialize()
{
    Super::Deinitialize();

    if (bHardFreezeActive)
    {
        UE_LOG(LogTemp, Warning, TEXT("Vajra was in hard freeze during shutdown"));
    }
}

void UVajraCircuitBreaker::InitializeTMRSystem(const FString& SatoshiSeed, float InitialPhi)
{
    SatoshiSeedHash = SatoshiSeed;
    CurrentPhi = InitialPhi;
    PhiWarningThreshold = InitialPhi;
    PhiCriticalThreshold = InitialPhi + 0.08f;

    // Initialize 3 TMR instances (Pattern I40)
    TMRInstances.Empty();
    for (int32 i = 0; i < 3; i++)
    {
        FTMRInstance Instance;
        Instance.InstanceID = i + 1;
        Instance.GravityVector = FVector(0, 0, -982.0f); // Standard UE gravity
        Instance.bIsHealthy = true;
        Instance.LastUpdateTime = FPlatformTime::Seconds();
        TMRInstances.Add(Instance);
    }

    UE_LOG(LogTemp, Log, TEXT("TMR System Initialized with %d instances"), TMRInstances.Num());
    UE_LOG(LogTemp, Log, TEXT("Satoshi Seed: %s"), *SatoshiSeedHash.Left(24));
}

void UVajraCircuitBreaker::SubmitPhysicsState(int32 InstanceID, const FTransform& AgentTransform,
                                             const FVector& CurrentGravity, const FVector& LinearVelocity)
{
    if (bHardFreezeActive) return;

    if (!TMRInstances.IsValidIndex(InstanceID - 1))
    {
        UE_LOG(LogTemp, Error, TEXT("Invalid Instance ID: %d"), InstanceID);
        return;
    }

    FTMRInstance& Instance = TMRInstances[InstanceID - 1];
    Instance.AgentPosition = AgentTransform.GetLocation();
    Instance.GravityVector = CurrentGravity;
    Instance.LastUpdateTime = FPlatformTime::Seconds();

    // Generate physics state hash
    FString StateString = FString::Printf(TEXT("%s|%s|%s|%f"),
        *Instance.AgentPosition.ToString(),
        *Instance.GravityVector.ToString(),
        *LinearVelocity.ToString(),
        Instance.LastUpdateTime);

    Instance.PhysicsHash = GenerateBlake3Hash(StateString);
}

FVajraStateSnapshot UVajraCircuitBreaker::ValidateTMRConsensus()
{
    FVajraStateSnapshot Snapshot;
    Snapshot.Timestamp = FPlatformTime::Cycles64();
    Snapshot.bIsCoherent = true;

    if (bHardFreezeActive)
    {
        Snapshot.AnomalyType = TEXT("HARD_FREEZE_ACTIVE");
        return Snapshot;
    }

    // Collect data from healthy instances
    TArray<FVector> Positions;
    TArray<FVector> Gravities;
    TArray<FString> Hashes;

    for (const FTMRInstance& Instance : TMRInstances)
    {
        if (Instance.bIsHealthy)
        {
            Positions.Add(Instance.AgentPosition);
            Gravities.Add(Instance.GravityVector);
            Hashes.Add(Instance.PhysicsHash);
        }
    }

    // Check for Byzantine faults via variance
    float PositionVariance = CalculateVariance(Positions);
    float GravityVariance = CalculateVariance(Gravities);

    // Calculate Phi based on system coherence
    Snapshot.Phi = 1.0f - FMath::Sqrt(PositionVariance + GravityVariance);
    Snapshot.Entropy = FMath::Log2(1.0f + PositionVariance * 1000.0f);

    // Check hash consensus (2/3 majority)
    TMap<FString, int32> HashCounts;
    for (const FString& Hash : Hashes)
    {
        HashCounts.FindOrAdd(Hash)++;
    }

    bool bHashConsensus = false;
    FString MajorityHash;
    for (const auto& Pair : HashCounts)
    {
        if (Pair.Value >= 2) // 2/3 consensus
        {
            bHashConsensus = true;
            MajorityHash = Pair.Key;
            break;
        }
    }

    // Detect Byzantine instances
    if (!bHashConsensus || PositionVariance > 0.000032f)
    {
        // Find faulty instance
        for (int32 i = 0; i < TMRInstances.Num(); i++)
        {
            if (TMRInstances[i].bIsHealthy && TMRInstances[i].PhysicsHash != MajorityHash)
            {
                IsolateFaultyInstance(i + 1);
                Snapshot.AnomalyType = FString::Printf(TEXT("BYZANTINE_INSTANCE_%d"), i + 1);
                Snapshot.bIsCoherent = false;
                OnByzantineDetected.Broadcast(i + 1, TEXT("TMR Consensus Failure"));
                break;
            }
        }
    }

    // Check Phi thresholds
    if (Snapshot.Phi >= PhiCriticalThreshold)
    {
        TriggerHardFreeze(FString::Printf(TEXT("PHI_CRITICAL_%f"), Snapshot.Phi), Snapshot.Phi);
        Snapshot.AnomalyType = TEXT("PHI_CRITICAL");
        Snapshot.bIsCoherent = false;
    }
    else if (Snapshot.Phi >= PhiWarningThreshold)
    {
        UE_LOG(LogTemp, Warning, TEXT("Phi Warning: %f"), Snapshot.Phi);
        Snapshot.AnomalyType = TEXT("PHI_WARNING");
    }

    // Update current Phi and notify
    if (CurrentPhi != Snapshot.Phi)
    {
        CurrentPhi = Snapshot.Phi;
        OnPhiChanged.Broadcast(CurrentPhi);
    }

    // Store in history
    StateHistory.Add(Snapshot);
    if (StateHistory.Num() > 1000) // Keep last 1000 snapshots
    {
        StateHistory.RemoveAt(0, StateHistory.Num() - 1000);
    }

    return Snapshot;
}

void UVajraCircuitBreaker::TriggerHardFreeze(const FString& Reason, float CurrentPhi)
{
    if (bHardFreezeActive) return;

    bHardFreezeActive = true;

    UE_LOG(LogTemp, Error, TEXT("=== VAJRA HARD FREEZE ACTIVATED ==="));
    UE_LOG(LogTemp, Error, TEXT("Reason: %s"), *Reason);
    UE_LOG(LogTemp, Error, TEXT("Phi: %f"), CurrentPhi);
    UE_LOG(LogTemp, Error, TEXT("Timestamp: %lld"), FPlatformTime::Cycles64());

    // 1. Freeze game simulation
    if (GEngine && GEngine->GetWorld())
    {
        GEngine->GetWorld()->GetWorldSettings()->SetTimeDilation(0.0f);
    }

    // 2. Seal current state to KARNAK
    FVajraStateSnapshot CurrentState;
    CurrentState.Phi = CurrentPhi;
    CurrentState.Timestamp = FPlatformTime::Cycles64();
    CurrentState.AnomalyType = Reason;
    SealStateToKarnak(TEXT("hard_freeze"), CurrentState);

    // 3. Notify SASC Cathedral
    NotifySASCathedral(TEXT("emergency_freeze"), CurrentState);

    // 4. Broadcast event
    OnHardFreeze.Broadcast(Reason);

    // 5. In production, this would trigger Kubernetes pod isolation
    // For now, we just log and freeze the game
    UE_LOG(LogTemp, Error, TEXT("Game frozen. Awaiting manual intervention."));
}

FString UVajraCircuitBreaker::GenerateBlake3Hash(const FString& Input)
{
    // NOTE: In full production, integrate the BLAKE3 C library here.
    // For standalone compilation, we use Unreal's built-in SHA1 as a robust fallback.
    FSHAHash Hash;
    FSHA1::HashBuffer(TCHAR_TO_ANSI(*Input), Input.Len(), Hash.Hash);
    return BytesToHex(Hash.Hash, 20);
}

float UVajraCircuitBreaker::CalculateVariance(const TArray<FVector>& Positions)
{
    if (Positions.Num() < 2) return 0.0f;

    // Calculate mean
    FVector Mean(0, 0, 0);
    for (const FVector& Pos : Positions)
    {
        Mean += Pos;
    }
    Mean /= Positions.Num();

    // Calculate variance
    float Variance = 0.0f;
    for (const FVector& Pos : Positions)
    {
        float Distance = FVector::Dist(Pos, Mean);
        Variance += Distance * Distance;
    }
    Variance /= Positions.Num();

    return Variance;
}

void UVajraCircuitBreaker::IsolateFaultyInstance(int32 InstanceID)
{
    if (TMRInstances.IsValidIndex(InstanceID - 1))
    {
        TMRInstances[InstanceID - 1].bIsHealthy = false;
        UE_LOG(LogTemp, Warning, TEXT("Isolated TMR Instance %d"), InstanceID);

        // Check if we still have quorum (2/3)
        int32 HealthyCount = 0;
        for (const FTMRInstance& Instance : TMRInstances)
        {
            if (Instance.bIsHealthy) HealthyCount++;
        }

        if (HealthyCount < 2)
        {
            TriggerHardFreeze(TEXT("TMR_QUORUM_LOST"), CurrentPhi);
        }
    }
}

void UVajraCircuitBreaker::SealStateToKarnak(const FString& SealType, const FVajraStateSnapshot& State)
{
    // Prepare JSON payload
    TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);
    JsonObject->SetStringField(TEXT("seal_type"), SealType);
    JsonObject->SetNumberField(TEXT("phi"), State.Phi);
    JsonObject->SetNumberField(TEXT("entropy"), State.Entropy);
    JsonObject->SetStringField(TEXT("anomaly_type"), State.AnomalyType);
    JsonObject->SetStringField(TEXT("satoshi_anchor"), SatoshiSeedHash);
    JsonObject->SetNumberField(TEXT("timestamp"), State.Timestamp);

    FString OutputString;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
    FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer);

    // Send to KARNAK sealer (HTTP POST)
    TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
    Request->SetURL(TEXT("http://localhost:9091/seal"));
    Request->SetVerb(TEXT("POST"));
    Request->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
    Request->SetContentAsString(OutputString);

    Request->OnProcessRequestComplete().BindLambda([](FHttpRequestPtr Request, FHttpResponsePtr Response, bool bSuccess)
    {
        if (bSuccess && Response.IsValid())
        {
            UE_LOG(LogTemp, Log, TEXT("KARNAK Seal successful: %s"), *Response->GetContentAsString());
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("KARNAK Seal failed"));
        }
    });

    Request->ProcessRequest();
}

void UVajraCircuitBreaker::UpdateSocialStress(float StressLevel, float CortisolDamping)
{
    // Apply Dor do Boto protocol (69% damping)
    float DampedStress = StressLevel * (1.0f - CortisolDamping);

    // Reduce game complexity based on stress
    if (DampedStress > 0.3f)
    {
        UE_LOG(LogTemp, Warning, TEXT("High social stress detected: %f"), DampedStress);
        UE_LOG(LogTemp, Warning, TEXT("Applying empathy damping: %f"), CortisolDamping);

        // In a real implementation, this would:
        // 1. Reduce NPC spawn rates
        // 2. Simplify AI behaviors
        // 3. Increase cooperation rewards
        // 4. Notify SASC for governance adjustments
    }
}
