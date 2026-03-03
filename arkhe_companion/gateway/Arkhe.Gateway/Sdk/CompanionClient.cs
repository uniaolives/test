using Grpc.Net.Client;
using System.Threading.Channels;
using Arkhe.Protos;

namespace Arkhe.Gateway.Sdk;

public class CompanionClient : IAsyncDisposable
{
    private readonly CompanionService.CompanionServiceClient _grpc;
    private readonly CancellationTokenSource _cts = new();

    public CompanionClient(string endpoint)
    {
        var channel = GrpcChannel.ForAddress(endpoint);
        _grpc = new CompanionService.CompanionServiceClient(channel);
    }

    public async Task<StateSnapshot> GetStateAsync()
    {
        return await _grpc.GetStateAsync(new StateRequest());
    }

    public async Task SetPhiAsync(double phi)
    {
        await _grpc.SetPhiAsync(new PhiRequest { Phi = phi });
    }

    public async ValueTask DisposeAsync()
    {
        _cts.Cancel();
    }
}
